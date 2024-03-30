import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from utils.system import *
from class_data.data import Data

class GenTfidf:
    def __init__(self,
                 query=None,
                 vector_data=None,
                 vector_column=None,
                 preprocess_data=None,
                 preprocess_column=None,
                 article_data=None,
                 tfidf=None,
                 method=None,
                 interval=None,
                 threshold=None,
                 ):

        '''
        query (str): User input (should contain a label, start date and end date)
        vector_data (pd.DataFrame): Pandas dataframe that stores the article vectors
        vector_column (str): Column name for the vector column
        preprocess_data (pd.DataFrame): Pandas dataframe that stores the preprocessed articles
        preprocess_column (str): Column name for the preprocess column
        article_data (pd.DataFrame): Pandas dataframe that stores the article headline and body text
        tfidf (tfidf.vectorizer): Fitted TFIDF Vectorizer
        method (str): Method to compute score for TFIDF (either 'mult' or 'cos_sim')
        interval (str): Date interval for generated index (either 'D' or 'M')
        threshold (int): Threshold parameter for transformation
        '''

        self.query = query
        self.vector_data = vector_data
        self.vector_column = vector_column
        self.preprocess_data = preprocess_data
        self.preprocess_column = preprocess_column
        self.article_data = article_data
        self.tfidf = tfidf
        self.method = method
        self.interval = interval
        self.threshold = threshold

        api_key = json.load(open(get_config() / 'api.json'))['openai_api_key']
        self.client = OpenAI(api_key=api_key)

        # TFIDF Extraction
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Here is the text: {self.query}"
                                            "Extract the label, start date, and end date from this piece of text. " +
                                            "If you cannot extract a label, start date, or end date, then store the them as 'none'. " +
                                            "Enhance the label by including details about what the label is and what is it about. " +
                                            "For example, if the label is 'ESG'. Enhance it by describing what ESG is and any information closely related to it " +
                                            "Make the label a descriptive, coherent paragraph and do not talk about the generated index. Only talk about the label meaning. " +
                                            "Next, create three different lists of length 3 each that store 3 unigrams, 3 bigrams, and 3 trigrams respectively of the enhanced label in all lowercase. " +
                                            "Your output should not contain any additional comments or add-ons. YOU MUST ONLY OUTPUT THIS: " +
                                            "{unigram: [(unigram1), ... , (unigram3)], " +
                                            "bigram: [(bigram1), ... , (bigram3)], " +
                                            "trigram: [(trigram1), ... , (trigram3)], "
                                            "start_date: (YYYY-MM-DD), " +
                                            "end_date: (YYYY-MM-DD)} "
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

        print(f"Here is the query: \n{self.query}")

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

        # Set timeframe
        if self.vector_data is not None and not self.vector_data.empty:
            self.vector_data = self.vector_data[(self.vector_data.index >= self.query["start_date"]) & (self.vector_data.index <= self.query["end_date"])]
        if self.preprocess_data is not None and not self.preprocess_data.empty:
            self.preprocess_data = self.preprocess_data[(self.preprocess_data.index >= self.query["start_date"]) & (self.preprocess_data.index <= self.query["end_date"])]
        if self.article_data is not None and not self.article_data.empty:
            self.article_data = self.article_data[(self.article_data.index >= self.query["start_date"]) & (self.article_data.index <= self.query["end_date"])]

    # Get query's tfidf vector
    def get_tfidf(self, ngram):
        return self.tfidf.transform([ngram])

    # Compare
    @staticmethod
    def compare_index(index, file_path):
        official_index = pd.read_parquet(get_format_data() / file_path)
        official_index.columns = ['official']
        index = index.join(official_index).dropna()
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_to_scale = index[['score', 'official']]
        scaled_data = scaler.fit_transform(data_to_scale)
        index[['score', 'official']] = scaled_data
        print("-" * 60)
        pearson_corr = index['score'].corr(index['official'], method='pearson')
        print(f"Pearson Correlation with EPU: {pearson_corr}")
        return index, pearson_corr

    # Save plot
    @staticmethod
    def exec_plot(prompt, label, pearson, data, names, output):
        # Make Dir
        plot_dir = f'../../view_attention/{output}'
        os.makedirs(plot_dir, exist_ok=True)

        # Save prompt and label
        with open(f'../../view_attention/{output}/{output}.txt', 'w') as file:
            file.write(f"Prompt: {prompt}\n")
            file.write(f"Label: {label}\n")
            file.write(f"Pearson: {pearson}\n")

        # Get plot
        plt.figure(figsize=(10, 5))
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'grey']

        # Adjust thickness of axis
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.3)

        # Plot each line
        for i, (column, name) in enumerate(zip(data.columns, names)):
            plt.plot(data.index, data[column], label=f"{name} Index", color=colors[i % len(colors)])

        # Set up legend, x-axis, y-axis, and grid
        plt.legend(fontsize='medium', prop={'weight': 'semibold'}, loc='upper left')
        plt.ylabel("Attention Index", color='black', fontweight='semibold', fontsize=12, labelpad=15)
        plt.xlabel("", color='black', fontweight='semibold', fontsize=12, labelpad=15)
        plt.xticks(color='black', fontweight='semibold')
        plt.tick_params(axis='x', which='major', direction='out', length=5, width=1.3, colors='black', labelsize=10, pad=5)
        plt.tick_params(axis='y', which='both', left=False, labelleft=False, pad=10)
        plt.grid(False)

        # Save figure
        plt.savefig(f'../../plot/{output}/{output}.jpg', format='jpg', dpi=300, bbox_inches='tight')
        plt.show()

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
            monthly_art = art.groupby(pd.Grouper(freq='M')).apply(lambda x: x.nlargest(1, 'score')).reset_index(level=0, drop=True)
            monthly_art.index = monthly_art.index.to_period('M').to_timestamp('M')

            # Join score and article
            sigmoid_score = sigmoid_score.resample('M').mean()
            sigmoid_score = pd.concat([sigmoid_score, monthly_art[['headline', 'body_txt']]], axis=1)

        elif self.interval == "D":
            daily_art = art.groupby(art.index.date).first()
            daily_art.index = pd.to_datetime(daily_art.index)
            # Join score and article
            sigmoid_score = pd.concat([sigmoid_score, daily_art[['headline', 'body_txt']]], axis=1)

        return sigmoid_score