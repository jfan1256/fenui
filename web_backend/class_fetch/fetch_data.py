import json
import numpy as np

from openai import OpenAI
from datetime import datetime, timedelta
from pymilvus import connections, Collection, utility
from sklearn.metrics.pairwise import cosine_similarity

from utils.system import *

class FetchData:
    def __init__(self,
                 label=None,
                 start_date=None,
                 end_date=None,
                 prod=None,
                 ngrok_host=None,
                 ngrok_port=None):

        '''
        label (str): Label to fetch for
        start_date (str: YYYY-MM-DD): Start date to fetch from
        end_date (str: YYYY-MM-DD): End date to fetch from
        '''

        self.label = label
        self.start_date = start_date
        self.end_date = end_date
        self.prod = prod
        self.ngrok_host = ngrok_host
        self.ngrok_port = ngrok_port

        # Get API Key
        api = json.load(open(get_config() / 'api.json'))
        self.api_key = api['openai_api_key']

        # Define OpenAI model
        self.model = 'text-embedding-ada-002'

        # Establish Milvus connection (Local or Prod)
        if self.prod:
            print("-" * 60 + f"\nConnecting to prod Milvus server")
            MILVUS_HOST = self.ngrok_host
            MILVUS_PORT = self.ngrok_port
            max_retries = 5
            attempts = 0

            while attempts < max_retries:
                try:
                    connections.disconnect('default')
                    connections.connect('default', host=MILVUS_HOST, port=MILVUS_PORT)
                    break
                except Exception as e:
                    attempts += 1
        else:
            print("-" * 60 + f"\nConnecting to local Milvus server")
            MILVUS_HOST = 'localhost'
            MILVUS_PORT = '19530'
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

        # Load collection
        collection = Collection("wsj_emb")
        collection.load()
        utility.load_state("wsj_emb")
        utility.loading_progress("wsj_emb")
        self.collection = collection

    # Convert date to int
    @staticmethod
    def _convert_date_to_int(date_str):
        return int(datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d"))

    # Convert int to date
    @staticmethod
    def _convert_int_to_date(date_int):
        date_str = str(date_int)
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        return date_obj.strftime("%Y-%m-%d")

    # Increment int date
    @staticmethod
    def _increment_date(date_str):
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        next_date_obj = date_obj + timedelta(days=1)
        next_date_str = next_date_obj.strftime("%Y-%m-%d")
        return next_date_str

    # Get OpenAI embedding
    def _get_openai_emb(self, text):
        client = OpenAI(api_key=self.api_key)
        embedding = client.embeddings.create(input=[text.replace("\n", " ")], model=self.model).data[0].embedding
        return embedding

    # # Fetch data from Milvus
    # def fetch_data(self):
    #     # Get label embedding
    #     label_emb = self._get_openai_emb(self.label)
    #
    #     # Set current date
    #     current_date = self.start_date
    #
    #     # Search Params
    #     search_params = {
    #         "metric_type": "COSINE",
    #         "params": {"nprobe": 100} # If nprobe is too small, it will not return "limit" articles
    #     }
    #
    #     # Prepare to store results
    #     all_results = []
    #
    #     # Iterate through each date
    #     while current_date <= self.end_date:
    #         print("-"*60)
    #         # Define query expression for specific date
    #         query_expr = f"date == '{current_date}'"
    #
    #         # Print number of articles in database for current date
    #         res = self.collection.query(
    #             expr=query_expr,
    #             output_fields=["count(*)"],
    #         )
    #         print(f"Number of articles in database for {current_date}: {res[0]['count(*)']}")
    #
    #         # Perform the search with date filtering
    #         results = self.collection.search(
    #             data=[label_emb],
    #             anns_field="embedding",
    #             param=search_params,
    #             limit=5,
    #             expr=query_expr,
    #             output_fields=['date', 'headline', 'document', 'n_date']
    #         )
    #
    #         # Check if there are results for the current date
    #         if results[0]:
    #             # Process results for the current date
    #             date_results = []
    #
    #             # Log data
    #             print(f"Number of results for {current_date} database: {len(results)}")
    #
    #             # Save data
    #             for hits in results:
    #                 for hit in hits:
    #                     processed_result = {
    #                         'id': hit.id,
    #                         'score': hit.score,
    #                         'date': hit.entity.get('date'),
    #                         'headline': hit.entity.get('headline'),
    #                         'document': hit.entity.get('document'),
    #                         'n_date': hit.entity.get('n_date')
    #                     }
    #                     date_results.append(processed_result)
    #
    #             all_results.append({current_date: date_results})
    #
    #         # Move to the next date
    #         current_date = self._increment_date(current_date)
    #
    #     return all_results

    # Fetch data from Milvus
    def fetch_data(self):
        # Get label embedding
        label_emb = self._get_openai_emb(self.label)
        label_emb = np.array(label_emb).reshape(1, -1)

        # Prepare to store results
        all_metadata = []
        all_embeddings = []

        # Initialize start and end dates for batching
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        batch_size = timedelta(days=90)
        count = 1

        print("Fetching Data...")
        while start_date < end_date:
            # Calculate the end date of the current batch
            current_end_date = min(start_date + batch_size, end_date)

            # Create query expression for the current batch
            query_expr = f"date >= {self._convert_date_to_int(start_date.strftime('%Y-%m-%d'))} and date <= {self._convert_date_to_int(current_end_date.strftime('%Y-%m-%d'))}"
            metadata = self.collection.query(expr=query_expr, output_fields=['date', 'embedding', 'headline', 'document'])
            embedding = self.collection.query(expr=query_expr, output_fields=['embedding'])

            print(f"Batch {count} ({start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}) - fetched with {len(metadata)} records.")

            # Process embeddings
            if embedding:
                batch_embeddings = [hit['embedding'] for hit in embedding]
                all_embeddings.extend(batch_embeddings)

            # Collect metadata for each hit in the batch
            for hit in metadata:
                collect_date = self._convert_int_to_date(hit.get('date'))
                collect_headline = hit.get('headline')
                collect_document = hit.get('document')
                all_metadata.append({'date': collect_date, 'headline': collect_headline, 'document': collect_document})

            # Move to the next batch
            start_date = current_end_date + timedelta(days=1)
            count+=1

        # Calculate score if embeddings were collected
        print("Calculating Scores...")
        if all_embeddings:
            embedding_matrix = np.vstack(all_embeddings)
            collect_score = cosine_similarity(label_emb, embedding_matrix)[0]
        else:
            collect_score = []

        # Create dict with all data
        data = {
            'date': [md['date'] for md in all_metadata],
            'cos_sim': collect_score,
            'headline': [md['headline'] for md in all_metadata],
            'document': [md['document'] for md in all_metadata]
        }

        return data