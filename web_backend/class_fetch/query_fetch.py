import time
import json

from openai import OpenAI
from datetime import datetime, timedelta
from pymilvus import connections, Collection, utility

from utils.system import *

class QueryFetch:
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
            MILVUS_HOST = 'localhost'
            MILVUS_PORT = '19530'
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        else:
            MILVUS_HOST = self.ngrok_host
            MILVUS_PORT = self.ngrok_port
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

        # Load collection
        collection = Collection("wsj_emb")
        collection.load()
        utility.load_state("wsj_emb")
        utility.loading_progress("wsj_emb")
        self.collection = collection

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

    # Query data from Milvus
    def query_fetch(self):
        # Get label embedding
        label_emb = self._get_openai_emb(self.label)

        # Set current date
        current_date = self.start_date

        # Search Params
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 100} # If nprobe is too small, it will not return "limit" articles
        }

        # Prepare to store results
        all_results = []

        # Iterate through each date
        while current_date <= self.end_date:
            # Define query expression for specific date
            query_expr = f"date == '{current_date}'"

            # # Print Count
            # res = self.collection.query(
            #     expr=query_expr,
            #     output_fields=["count(*)"],
            # )
            # print(res[0])


            # Perform the search with date filtering
            results = self.collection.search(
                data=[label_emb],
                anns_field="embedding",
                param=search_params,
                limit=5,
                expr=query_expr,
                output_fields=['date', 'headline', 'document', 'n_date']
            )

            # Check if there are results for the current date
            if results[0]:
                # Process results for the current date
                date_results = []
                for hits in results:
                    for hit in hits:
                        processed_result = {
                            'id': hit.id,
                            'score': hit.score,
                            'date': hit.entity.get('date'),
                            'headline': hit.entity.get('headline'),
                            'document': hit.entity.get('document'),
                            'n_date': hit.entity.get('n_date')
                        }
                        date_results.append(processed_result)

                all_results.append({current_date: date_results})

            # Move to the next date
            current_date = self._increment_date(current_date)

        return all_results