import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils.system import *
from class_data.data import Data

class GenIndex:
    def __init__(self,
                 query=None,
                 type=None,
                 vector_data=None,
                 vector_column=None,
                 preprocess_data=None,
                 preprocess_column=None,
                 article_data=None,
                 tfidf=None,
                 method=None,
                 interval=None,
                 threshold=None,
                 alpha=None,
                 ):

        '''
        query (str): User input (should contain a label, start date and end date)
        type (str): Method to generate an index (either 'embedding', 'tfidf', 'count')
        vector_data (pd.DataFrame): Pandas dataframe that stores the article vectors
        vector_column (str): Column name for the vector column
        preprocess_data (pd.DataFrame): Pandas dataframe that stores the preprocessed articles
        preprocess_column (str): Column name for the preprocess column
        article_data (pd.DataFrame): Pandas dataframe that stores the article headline and body text
        tfidf (tfidf.vectorizer): Fitted TFIDF Vectorizer
        method (str): Method to compute score for TFIDF (either 'mult' or 'cos_sim')
        interval (str): Date interval for generated index (either 'D' or 'M')
        threshold (int): Threshold parameter for transformation
        alpha (float): Time decay alpha
        '''

        self.query = query
        self.type = type
        self.vector_data = vector_data
        self.vector_column = vector_column
        self.preprocess_data = preprocess_data
        self.preprocess_column = preprocess_column
        self.article_data = article_data
        self.tfidf = tfidf
        self.method = method
        self.interval = interval
        self.threshold = threshold
        self.alpha = alpha

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
                                                "Make the label a descriptive, coherent paragraph and do not talk about the generated index. Only talk about the label meaning. " +
                                                "If you cannot extract a label, start date, or end date, then store the them as 'none'. " +
                                                "Your output should not contain any additional comments or add-ons. YOU MUST ONLY OUTPUT THIS: " +
                                                "\n\n{\"label\": \"(enhanced_label)\", " +
                                                "\"start_date\": \"(YYYY-MM-DD)\", " +
                                                "\"end_date\": \"(YYYY-MM-DD)\"} "
                     }
                ],
                temperature=1,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                seed=123
            )
            summary = response.choices[0].message.content.strip()
            start_index = summary.find('{')
            summary = summary[start_index:]
            end_index = summary.rfind('}') + 1
            summary = summary[:end_index]
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
                                                "Make the label a descriptive, coherent paragraph and do not talk about the generated index. Only talk about the label meaning. " +
                                                "Next, create three different lists of length 3 each that store 3 unigrams, 3 bigrams, and 3 trigrams respectively of the enhanced label in all lowercase. " +
                                                "\n\n Your output should not contain any additional comments or add-ons. YOU MUST ONLY OUTPUT THIS: " +
                                                "{\"unigram\": [\"(unigram1)\", ... , \"(unigram3)\"], " +
                                                "\"bigram\": [\"(bigram1)\", ... , \"(bigram3)\"], " +
                                                "\"trigram\": [\"(trigram1)\", ... , \"(trigram3)\"], "
                                                "\"start_date\": \"(YYYY-MM-DD)\", " +
                                                "\"end_date\": \"(YYYY-MM-DD)\"} "
                     }
                ],
                temperature=1,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                seed=123
            )
            summary = response.choices[0].message.content.strip()
            start_index = summary.find('{')
            summary = summary[start_index:]
            end_index = summary.rfind('}') + 1
            summary = summary[:end_index]
            self.query = json.loads(summary)

        # Count Extraction
        elif self.type == "count":
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": f"Here is the text: {self.query}" +
                                                "\n\n :: " +
                                                "Extract the label, start date, and end date from this piece of text and make it all lowercase. " +
                                                "If you cannot extract a label, start date, or end date, then store the them as 'none'. " +
                                                "\n\n Your output should not contain any additional comments or add-ons. YOU MUST ONLY OUTPUT THIS: " +
                                                "{\"label\": \"(label)\", " +
                                                "\"start_date\": \"(YYYY-MM-DD)\", " +
                                                "\"end_date\": \"(YYYY-MM-DD)\"} "
                     }
                ],
                temperature = 1,
                max_tokens = 150,
                top_p = 1,
                frequency_penalty = 0,
                presence_penalty = 0,
                seed = 123
            )
            summary = response.choices[0].message.content.strip()
            start_index = summary.find('{')
            summary = summary[start_index:]
            end_index = summary.rfind('}') + 1
            summary = summary[:end_index]
            self.query = json.loads(summary)

        print(f"Here is the query: \n{self.query}")

        # Set timeframe
        if self.vector_data is not None and not self.vector_data.empty:
            self.vector_data = self.vector_data[(self.vector_data.index >= self.query["start_date"]) & (self.vector_data.index <= self.query["end_date"])]
        if self.preprocess_data is not None and not self.preprocess_data.empty:
            self.preprocess_data = self.preprocess_data[(self.preprocess_data.index >= self.query["start_date"]) & (self.preprocess_data.index <= self.query["end_date"])]
        if self.article_data is not None and not self.article_data.empty:
            self.article_data = self.article_data[(self.article_data.index >= self.query["start_date"]) & (self.article_data.index <= self.query["end_date"])]

        # Set limit for number of articles per date
        if self.vector_data is not None and not self.vector_data.empty:
            limit = 30
            count = self.vector_data.groupby(self.vector_data.index)[self.vector_data.columns[0]].count()
            valid_date = count >= limit
            self.vector_data = self.vector_data[self.vector_data.index.isin(count[valid_date].index)]
        if self.preprocess_data is not None and not self.preprocess_data.empty:
            limit = 30
            count = self.preprocess_data.groupby(self.preprocess_data.index)[self.preprocess_data.columns[0]].count()
            valid_date = count >= limit
            self.preprocess_data = self.preprocess_data[self.preprocess_data.index.isin(count[valid_date].index)]
        if self.article_data is not None and not self.article_data.empty:
            limit = 30
            count = self.article_data.groupby(self.article_data.index)[self.article_data.columns[0]].count()
            valid_date = count >= limit
            self.article_data = self.article_data[self.article_data.index.isin(count[valid_date].index)]

    # Get query's openai embedding
    def get_emb(self):
        text = self.query['label'].replace("\n", " ")
        return self.client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding

    # Get query's tfidf vector
    def get_tfidf(self, ngram):
        return self.tfidf.transform([ngram])

    # Time decay
    @staticmethod
    def time_decay(series, alpha):
        end_of_month = series.index.to_period('M').to_timestamp('M') + pd.offsets.MonthEnd(1)
        days_diff = (end_of_month - series.index).days
        decay_factors = np.exp(-alpha * days_diff)
        decayed_scores = series * decay_factors
        return decayed_scores

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
        relu_score = np.maximum(0, self.vector_data['score'] - self.threshold)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------AGGREGATE--------------------------------------------------------------------
        # Add article
        relu_score.index.names = ['date']

        # Apply time decay
        # relu_score = self.time_decay(relu_score, self.alpha)
        relu_score = relu_score.to_frame('score')

        # Get articles
        art = pd.concat([relu_score, self.article_data[['headline', 'body_txt']]], axis=1)
        art = art.sort_values(by='score', ascending=False)

        # Aggregate to daily timeframe
        relu_score = relu_score.groupby('date').mean()
        relu_score.columns = ['score']

        # Aggregate to monthly timeframe or daily timeframe
        if self.interval == "M":
            # # Retrieve top article per month
            monthly_art = art.resample('M').first()

            # Join score and article
            relu_score = relu_score.resample('M').mean()
            relu_score = pd.concat([relu_score, monthly_art[['headline', 'body_txt']]], axis=1)

            # # Aggregate decayed scores and normalize by sum of decay factors
            # relu_score = relu_score.resample('M').sum()
            # # Retrieve top article per month (consider adjusting this part to align with decayed scores logic)
            # monthly_art = art.resample('M').first()
            # # Join decayed average score and article
            # relu_score = pd.concat([relu_score, monthly_art[['headline', 'body_txt']]], axis=1)

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
        batch_size = 10000
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

                for batch_num in tqdm(range(num_batches), desc="Processing batches..."):
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
        # Apply Sigmoid transformation
        score = self.vector_data[self.vector_data.columns[1:]]
        sigmoid_score = 1 / (1 + np.exp(-(self.threshold * score + -1)))
        sigmoid_score = np.maximum(0, score - self.threshold)
        sigmoid_score = sigmoid_score.mean(axis=1).to_frame("score")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------AGGREGATE---------------------------------------------------------------------
        # Add article
        sigmoid_score.index.names = ['date']
        art = pd.concat([sigmoid_score, self.article_data[['headline', 'body_txt']]], axis=1)
        art = art.sort_values(by='score', ascending=False)

        # Aggregate to daily timeframe
        sigmoid_score = sigmoid_score.groupby('date').mean()
        sigmoid_score.columns = ['score']

        # Aggregate to monthly timeframe or daily timeframe
        if self.interval == "M":
            # Retrieve top article per month
            monthly_art = art.resample('M').first()

            # Join score and article
            sigmoid_score = sigmoid_score.resample('M').mean()
            sigmoid_score = pd.concat([sigmoid_score, monthly_art[['headline', 'body_txt']]], axis=1)

        elif self.interval == "D":
            daily_art = art.groupby(art.index.date).first()
            daily_art.index = pd.to_datetime(daily_art.index)
            # Join score and article
            sigmoid_score = pd.concat([sigmoid_score, daily_art[['headline', 'body_txt']]], axis=1)

        return sigmoid_score

    # Generate Count Index
    def generate_count(self):
        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------COMPUTE SCORE-------------------------------------------------------------------
        print("-" * 60 + "\nComputing score...")
        # Compute score
        self.preprocess_data['score'] = self.preprocess_data[self.preprocess_column].str.count(self.query['label'])
        score = self.preprocess_data[['score']]

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------AGGREGATE---------------------------------------------------------------------
        # Add article
        art = pd.concat([score, self.article_data[['headline', 'body_txt']]], axis=1)
        art = art.sort_values(by='score', ascending=False)

        # Aggregate to daily timeframe
        score = score.groupby('date').sum()
        score.columns = ['score']

        # Aggregate to monthly timeframe or daily timeframe
        if self.interval == "M":
            # Retrieve top article per month
            monthly_art = art.resample('M').first()

            # Join score and article
            score = score.resample('M').mean()
            score = pd.concat([score, monthly_art[['headline', 'body_txt']]], axis=1)

        elif self.interval == "D":
            daily_art = art.groupby(art.index.date).first()
            daily_art.index = pd.to_datetime(daily_art.index)
            # Join score and article
            score = pd.concat([score, daily_art[['headline', 'body_txt']]], axis=1)

        return score

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------TEST--------------------------------------------------------------------------
if __name__ == "__main__":
    # Load openai embeddings
    wsj_openai = Data(folder_path=get_format_data() / 'openai', file_pattern='wsj_emb_openai_*')
    wsj_openai = wsj_openai.concat_files(10)

    # Load articles
    wsj_art = Data(folder_path=get_format_data() / 'token', file_pattern='wsj_tokens_*')
    wsj_art = wsj_art.concat_files(1)

    # Equal data
    wsj_openai = wsj_openai.head(10000)
    wsj_art = wsj_art.head(10000)

    # Params
    type = 'embedding'
    vector_column = 'ada_embedding'
    interval = 'M'

    # Generate
    query = 'Generate an index with label ESG from 1984-01-02 to 2021-12-31'
    generate = GenIndex(query=query,
                        type=type,
                        vector_data=wsj_openai,
                        vector_column=vector_column,
                        article_data=wsj_art,
                        interval=interval,
                        threshold=0.77,
                        alpha=0.01)
    esg = generate.generate_emb()
    esg.plot()
    plt.title('ESG Index Over Time')
    plt.xlabel('Date')
    plt.ylabel('ESG Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()