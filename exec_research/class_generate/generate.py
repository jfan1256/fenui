import json
import numpy as np
import pandas as pd

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from utils.system import *
from class_data.data import Data

class Generate:
    def __init__(self,
                 query=None,
                 type=None,
                 vector_data=None,
                 vector_column=None,
                 article_data=None,
                 tfidf=None,
                 method=None,
                 interval=None
                 ):

        '''
        query (str): User input (should contain a label, start date and end date)
        type (str): Method to generate an index (either 'embedding' or 'tfidf')
        vector_data (pd.DataFrame): Pandas dataframe that stores the article vectors
        vector_column (str): Column name for the vector column
        article_data (pd.DataFrame): Pandas dataframe that stores the article headline and body text
        tfidf (tfidf.vectorizer): Fitted TFIDF Vectorizer
        method (str): Method to compute score for TFIDF (either 'mult' or 'cos_sim')
        interval (str): Date interval for generated index (either 'D' or 'M')
        '''

        self.query = query
        self.type = type
        self.vector_data = vector_data
        self.vector_column = vector_column
        self.article_data = article_data
        self.tfidf = tfidf
        self.method = method
        self.interval = interval

        api_key = json.load(open(get_config() / 'api.json'))['openai_api_key']
        self.client = OpenAI(api_key=api_key)

        # Embedding Extraction
        if self.type == "embedding":
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": f"Here is the text: {self.query}" +
                                                "\n\n :: " +
                                                "Extract the label, start date, and end date from this piece of text. " +
                                                "Enhance the label by including details about what the label is and what is it about. " +
                                                "For example, if the label is 'ESG'. Enhance it by describing what ESG is and any information closely related to it " +
                                                "Make the label a coherent paragraph of max three sentences and do not talk about the generated index. Only talk about the label meaning. " +
                                                "If you cannot extract a label, start date, or end date, then store the them as 'none'. " +
                                                "Your output should not contain any additional comments or add-ons. YOU MUST ONLY OUTPUT THIS: " +
                                                "\n\n{\"label\": \"(enhanced_label)\", " +
                                                "\"start_date\": \"(YYYY-MM-DD)\", " +
                                                "\"end_date\": \"(YYYY-MM-DD)\"} "
                     }
                ]
            )
            summary = response.choices[0].message.content.strip()
            start_index = summary.find('{')
            summary = summary[start_index:]
            self.query = json.loads(summary)

        # TFIDF Extraction
        elif self.type == "tfidf":
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": f"Here is the text: {self.query}" +
                                                "\n\n :: " +
                                                "Extract the label, start date, and end date from this piece of text. " +
                                                "If you cannot extract a label, start date, or end date, then store the them as 'none'. " +
                                                "Enhance the label by including details about what the label is and what is it about. " +
                                                "For example, if the label is 'ESG'. Enhance it by describing what ESG is and any information closely related to it " +
                                                "Make the label a coherent paragraph of max three sentences and do not talk about the generated index. Only talk about the label meaning. " +
                                                "Next, create three different lists of length 3 each that store 3 unigrams, 3 bigrams, and 3 trigrams respectively of the enhanced label in all lowercase. " +
                                                "\n\n Your output should not contain any additional comments or add-ons. YOU MUST ONLY OUTPUT THIS: " +
                                                "{\"unigram\": [\"(unigram1)\", ... , \"(unigram3)\"], " +
                                                "\"bigram\": [\"(bigram1)\", ... , \"(bigram3)\"], " +
                                                "\"trigram\": [\"(trigram1)\", ... , \"(trigram3)\"], "
                                                "\"start_date\": \"(YYYY-MM-DD)\", " +
                                                "\"end_date\": \"(YYYY-MM-DD)\"} "
                     }
                ]
            )
            summary = response.choices[0].message.content.strip()
            start_index = summary.find('{')
            summary = summary[start_index:]
            self.query = json.loads(summary)

        print(f"Here is the query: \n{self.query}")

        # Set timeframe
        self.vector_data = self.vector_data[(self.vector_data.index >= self.query["start_date"]) & (self.vector_data.index <= self.query["end_date"])]
        self.article_data = self.article_data[(self.article_data.index >= self.query["start_date"]) & (self.article_data.index <= self.query["end_date"])]

        # Set limit for number of articles per date
        limit = 30
        count = self.vector_data.groupby(self.vector_data.index)[self.vector_data.columns[0]].count()
        valid_date = count >= limit

        self.vector_data = self.vector_data[self.vector_data.index.isin(count[valid_date].index)]
        self.article_data = self.article_data[self.article_data.index.isin(count[valid_date].index)]

    # Get query's openai embedding
    def get_emb(self):
        text = self.query['label'].replace("\n", " ")
        return self.client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding

    # Get query's tfidf vector
    def get_tfidf(self, ngram):
        return self.tfidf.transform([ngram])

    # Generate Embedding Index
    def generate_emb(self):
        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------GET QUERY VECTOR------------------------------------------------------------------
        print("-"*60 + "\nGetting query vector...")
        label_emb = self.get_emb()
        label_emb = np.array(label_emb).reshape(1, -1)

        # Convert vector_data to a matrix
        vector_matrix = np.stack(self.vector_data[self.vector_column].values)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------COMPUTE COS SIM------------------------------------------------------------------
        print("-"*60 + "\nComputing cosine similarity with label embedding...")
        cos_sim = cosine_similarity(label_emb, vector_matrix)
        self.vector_data['score'] = cos_sim[0]

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------APPLY TRANSFORMATION---------------------------------------------------------------
        # Apply RELU transformation
        relu_score = np.maximum(0, self.vector_data['score'] - 0.75)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------AGGREGATE--------------------------------------------------------------------
        # Add article
        relu_score.index.names = ['date']
        art = pd.concat([relu_score, self.article_data[['headline', 'body_txt']]], axis=1)
        art = art.sort_values(by='score', ascending=False)

        # Aggregate to daily timeframe
        relu_score = relu_score.groupby('date').mean()
        relu_score.columns = ['score']

        # Aggregate to monthly timeframe or daily timeframe
        if self.interval == "M":
            # Retrieve top article per month
            monthly_art = art.resample('M').first()

            # Join score and article
            relu_score = relu_score.resample('M').mean()
            relu_score = pd.concat([relu_score, monthly_art[['headline', 'body_txt']]], axis=1)

        elif self.interval == "D":
            daily_art = art.groupby(art.index.date).first()
            daily_art.index = pd.to_datetime(daily_art.index)
            # Join score and article
            relu_score = pd.concat([relu_score, daily_art[['headline', 'body_txt']]], axis=1)

        return relu_score

    # Generate TFIDF Index
    def generate_tfidf(self):
        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------COMPUTE SCORE-------------------------------------------------------------------
        print("-"*60 + "\nComputing score...")
        # Compute score across ngrams
        batch_size = 100000
        count = 0

        for key in self.query.keys():
            # Skip over keys: start_date and end_date
            if key == "start_date" or key == "end_date":
                break

            for ngram in self.query[key]:
                print("-" * 60 + f"\nProcessing ngram: {ngram}")
                ngram_tfidf = self.get_tfidf(ngram)

                # Compute score in batches
                num_batches = len(self.vector_data) // batch_size + 1
                comp_batches = []

                for batch_num in range(num_batches):
                    start_idx = batch_num * batch_size
                    end_idx = min((batch_num + 1) * batch_size, len(self.vector_data))
                    matrix_batch = np.stack(self.vector_data[self.vector_column][start_idx:end_idx].values)

                    if self.method == 'mult':
                        # Compute matrix multiplication
                        comp_batch = ngram_tfidf @ matrix_batch.T
                        comp_batches.append(comp_batch)

                    elif self.method == 'cos_sim':
                        # Compute cosine similarity
                        comp_batch = cosine_similarity(ngram_tfidf, matrix_batch)
                        comp_batches.append(comp_batch)

                # Concatenate the cosine similarity batches
                score = np.concatenate([arr[0] for arr in comp_batches])

                # Create column name with key
                column_name = f'score_{count}'
                self.vector_data[column_name] = score
                count += 1

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------APPLY TRANSFORMATION---------------------------------------------------------------
        # Apply RELU transformation
        score = self.vector_data[self.vector_data.columns[1:]]
        relu_score = np.maximum(0, score - 0.75)
        relu_score = relu_score.mean(axis=1).to_frame("score")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------AGGREGATE---------------------------------------------------------------------
        # Add article
        relu_score.index.names = ['date']
        art = pd.concat([relu_score, self.article_data[['headline', 'body_txt']]], axis=1)
        art = art.sort_values(by='score', ascending=False)

        # Aggregate to daily timeframe
        relu_score = relu_score.groupby('date').mean()
        relu_score.columns = ['score']

        # Aggregate to monthly timeframe or daily timeframe
        if self.interval == "M":
            # Retrieve top article per month
            monthly_art = art.resample('M').first()

            # Join score and article
            relu_score = relu_score.resample('M').mean()
            relu_score = pd.concat([relu_score, monthly_art[['headline', 'body_txt']]], axis=1)

        elif self.interval == "D":
            daily_art = art.groupby(art.index.date).first()
            daily_art.index = pd.to_datetime(daily_art.index)
            # Join score and article
            relu_score = pd.concat([relu_score, daily_art[['headline', 'body_txt']]], axis=1)

        return relu_score

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------TEST--------------------------------------------------------------------------
if __name__ == "__main__":
    # Load openai embeddings
    wsj_openai = Data(folder_path=get_format_data() / 'openai', file_pattern='wsj_emb_openai_*')
    wsj_openai = wsj_openai.concat_files(3)

    # Load articles
    wsj_art = Data(folder_path=get_format_data() / 'token', file_pattern='wsj_tokens_*')
    wsj_art = wsj_art.concat_files(3)

    # Query
    query = 'Generate an index with label ESG from 1984-01-02 to 1984-01-23'
    type = 'embedding'
    vector_column = 'ada_embedding'
    interval = "M"

    # Generate
    generate = Generate(query=query, type=type, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval="M")
    index = generate.generate_emb()
