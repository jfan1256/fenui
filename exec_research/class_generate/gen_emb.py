import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from utils.system import *
from class_data.data import Data

class GenEmb:
    def __init__(self,
                 query=None,
                 type=None,
                 vector_data=None,
                 vector_column=None,
                 preprocess_data=None,
                 preprocess_column=None,
                 article_data=None,
                 interval=None,
                 threshold=None,
                 ):

        '''
        query (str): User input (should contain a label, start date and end date)
        type (str): Method to generate an index (either 'embedding', 'tfidf', 'count')
        vector_data (pd.DataFrame): Pandas dataframe that stores the article vectors
        vector_column (str): Column name for the vector column
        preprocess_data (pd.DataFrame): Pandas dataframe that stores the preprocessed articles
        preprocess_column (str): Column name for the preprocess column
        article_data (pd.DataFrame): Pandas dataframe that stores the article headline and body text
        interval (str): Date interval for generated index (either 'D' or 'M')
        threshold (int): Threshold parameter for transformation
        '''

        self.query = query
        self.type = type
        self.vector_data = vector_data
        self.vector_column = vector_column
        self.preprocess_data = preprocess_data
        self.preprocess_column = preprocess_column
        self.article_data = article_data
        self.interval = interval
        self.threshold = threshold

        api_key = json.load(open(get_config() / 'api.json'))['openai_api_key']
        self.client = OpenAI(api_key=api_key)

        # Embedding Extraction
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": "Given a user's textual request about tracking the attention index of a specific topic or narrative over time, extract the relevant fields so that they can be supplied to function in a structured format."
                                            f"Here is the user’s request input: {self.query}"
                                            "The fields to extract are:"
                                            "query: The specific topic or narrative the user wants to track. If not specified, return null."
                                            "start_date: The start date for tracking, formatted as YYYY-MM-DD. If not specified, return null."
                                            "end_date: The end date for tracking, formatted as YYYY-MM-DD. If not specified, return null."
                                            "Return a JSON object with the keys query, start_date, and end_date. If a field is missing, set the corresponding value to null. This will indicate that the field was not specified by the user and can be filled with default values later."
                 }
            ],
            temperature=0.75,
            max_tokens=500,
            top_p=0.75,
            frequency_penalty=0,
            presence_penalty=0,
            seed=1
        )
        summary = response.choices[0].message.content.strip()
        self.query = json.loads(summary)

        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": "Your job is to output expanded queries to represent the concept in the input query:"
                                            f"'{self.query['query']}'"
                                            "In detail, I have over 800000 news articles."
                                            "I want to track how much each article pertains to the a topic or a narrative contained in the input query:"
                                            f"'{self.query['query']}'"
                                            "I am going to use openai's embedding model to compare the cosine similarity between each article and the list of expanded queries that encapsulate the concept in the input query."
                                            "Please transform the input query into an extensive set of queries that accurately, thoroughly, and vividly encompass all interpretations, perspectives, and facets of the input query."
                                            "Please output in JSON format as {expanded_queries: [expanded_query_1, expanded_query_2, expanded_query_3, …]}."
                 }
            ],
            temperature=0.75,
            max_tokens=500,
            top_p=0.75,
            frequency_penalty=0,
            presence_penalty=0,
            seed=1
        )
        summary = response.choices[0].message.content.strip()
        summary = json.loads(summary)
        paragraph = '. '.join(summary['expanded_queries']) + '.'
        self.query['query'] = paragraph

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

    # Get query's openai embedding
    def get_emb(self):
        text = self.query['query'].replace("\n", " ")
        return self.client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

    # Compare
    @staticmethod
    def compare_index(index, file_path):
        official_index = pd.read_parquet(get_format_data() / file_path)
        official_index.columns = ['official']
        if not official_index.index.freqstr == 'M':
            official_index = official_index.resample('M').mean()
        index = index.join(official_index).dropna()
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_to_scale = index[['score', 'official']]
        scaled_data = scaler.fit_transform(data_to_scale)
        index[['score', 'official']] = scaled_data
        print("-" * 60)
        pearson_corr = index['score'].corr(index['official'], method='pearson')
        print(f"Pearson Correlation with EPU: {pearson_corr}")
        return index, pearson_corr

    # Join
    @staticmethod
    def join_index(index, file_path):
        official_index = pd.read_parquet(get_format_data() / file_path)
        official_index.columns = ['official']
        official_index = pd.read_parquet(get_format_data() / file_path)
        official_index.columns = ['official']
        if not official_index.index.freqstr == 'M':
            official_index = official_index.resample('M').mean()
        index = index.join(official_index)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_to_scale = index[['score', 'official']]
        scaled_data = scaler.fit_transform(data_to_scale)
        index[['score', 'official']] = scaled_data
        return index

    # Save plot
    @staticmethod
    def exec_plot(prompt, label, pearson, data, names, output):
        # Make Dir
        plot_dir = f'../../plot/{output}'
        os.makedirs(plot_dir, exist_ok=True)

        # Save prompt and label
        with open(f'../../plot/{output}/{output}.txt', 'w') as file:
            file.write(f"Prompt: {prompt}\n\n")
            file.write(f"Label: {label}\n\n")
            file.write(f"Pearson: {pearson}\n\n")

        # Get plot
        plt.figure(figsize=(10, 5))
        colors = ['blue', 'red', 'cyan', 'orange', 'green', 'purple', 'magenta', 'yellow', 'black', 'grey']

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
        relu_score = relu_score.to_frame('score')

        # Get articles
        art = pd.concat([relu_score, self.article_data[['headline', 'body_txt']]], axis=1)
        art = art.sort_values(by='score', ascending=False)

        # Aggregate to daily timeframe
        relu_score = relu_score.groupby('date').mean()
        relu_score.columns = ['score']

        # Aggregate to monthly timeframe or daily timeframe
        if self.interval == "M":
            # Retrieve top article per month
            monthly_art = art.groupby(pd.Grouper(freq='M')).apply(lambda x: x.nlargest(1, 'score')).reset_index(level=0, drop=True)
            monthly_art.index = monthly_art.index.to_period('M').to_timestamp('M')

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
# --------------------------------------------------------------------------GET PLOTS---------------------------------------------------------------------
if __name__ == "__main__":
    # # Load openai embeddings
    # wsj_openai = Data(folder_path=get_format_data() / 'openai', file_pattern='wsj_emb_openai_*')
    # wsj_openai = wsj_openai.concat_files(10)

    # # Load articles
    # wsj_art = Data(folder_path=get_format_data() / 'token', file_pattern='wsj_tokens_*')
    # wsj_art = wsj_art.concat_files(1)

    # # Equal data
    # wsj_openai = wsj_openai.head(10000)
    # wsj_art = wsj_art.head(10000)

    # Load openai embeddings
    wsj_openai = Data(folder_path=get_format_data() / 'openai', file_pattern='wsj_emb_textemb3small_*')
    wsj_openai = wsj_openai.concat_files()

    # Load articles
    wsj_art = Data(folder_path=get_format_data() / 'token', file_pattern='wsj_tokens_*')
    wsj_art = wsj_art.concat_files()

    # Params
    vector_column = 'ada_embedding'
    interval = 'M'
    threshold = 0.40

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-"*120)
    query = 'Generate an index with label ESG from January 1st, 1984, to December 31st, 2021.'
    generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    index = generate.generate_emb()
    index = generate.join_index(index=index, file_path='esg_google_trend.parquet.brotli')
    generate.exec_plot(prompt=query, label=generate.query['query'], pearson=0, data=index[['score', 'official']], names=['ESG', 'ESG (Google Trend)'], output='esg_index')

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-"*120)
    query = 'Generate an index with label US Economic Policy Uncertainty from January 1st, 1984, to December 31st, 2021.'
    generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    index = generate.generate_emb()
    index, pearson_corr = generate.compare_index(index=index, file_path='epu.parquet.brotli')
    generate.exec_plot(prompt=query, label=generate.query['query'], pearson=pearson_corr, data=index[['score', 'official']], names=['US EPU', 'US EPU (Baker et al.)'], output='usepu_index')

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-" * 120)
    query = 'Generate an index with label Systemic Financial Stress from January 1st, 1984, to December 31st, 2021.'
    generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    index = generate.generate_emb()
    index, pearson_corr = generate.compare_index(index=index, file_path='fsi.parquet.brotli')
    generate.exec_plot(prompt=query, label=generate.query['query'], pearson=pearson_corr, data=index[['score', 'official']], names=['Financial Stress', 'Financial Stress (Baker et al.)'], output='usepu_index')

    # window = 6
    # index[f'score{window}'] = index['score'].rolling(window).mean()
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scale_score_ma = scaler.fit_transform(index[f'score{window}'].values.reshape(-1, 1))
    # index[f'score{window}'] = scale_score_ma
    # index, pearson_corr = generate.compare_index(index=index, file_path='fsi.parquet.brotli')
    # index_ma = index[[f'score{window}']]
    # index_ma.columns = ['score']
    # index_ma, pearson_corr_ma = generate.compare_index(index=index_ma, file_path='fsi.parquet.brotli')
    # pearson_corr = [pearson_corr, pearson_corr_ma]
    # generate.exec_plot(prompt=query, label=generate.query['query'], pearson=pearson_corr, data=index[['score', 'official', f'score{window}']], names=['Financial Stress', 'Financial Stress (OFR)', f'Financial Stress {window} Month MA'], output='fsi_index')

    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-"*120)
    # query = 'Generate an index with label US-China Trade War from January 1st, 1984, to December 31st, 2021.'
    # generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    # index = generate.generate_emb()
    # generate.exec_plot(prompt=query, label=generate.query['query'], pearson=0, data=index[['score']], names=['US-China Trade War'], output='uschinatradewar_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-"*120)
    # query = 'Generate an index with label Artificial Intelligence from January 1st, 1984, to December 31st, 2021.'
    # generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    # index = generate.generate_emb()
    # generate.exec_plot(prompt=query, label=generate.query['query'], pearson=0, data=index[['score']], names=['Artificial Intelligence'], output='ai_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-"*120)
    # query = 'Generate an index with label Blockchain from January 1st, 1984, to December 31st, 2021.'
    # generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    # index = generate.generate_emb()
    # generate.exec_plot(prompt=query, label=generate.query['query'], pearson=0, data=index[['score']], names=['Blockchain'], output='blockchain_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-"*120)
    # query = 'Generate an index with label COVID-19 from January 1st, 1984, to December 31st, 2021.'
    # generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    # index = generate.generate_emb()
    # generate.exec_plot(prompt=query, label=generate.query['query'], pearson=0, data=index[['score']], names=['COVID-19'], output='covid19_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-"*120)
    # query = 'Generate an index with label Economic Recession from January 1st, 1984, to December 31st, 2021.'
    # generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    # index = generate.generate_emb()
    # index, pearson_corr = generate.compare_index(index=index, file_path='recession.parquet.brotli')
    # generate.exec_plot(prompt=query, label=generate.query['query'], pearson=pearson_corr, data=index[['score', 'official']], names=['Economic Recession', 'Economic Recession (Bybee et al.)'], output='economicrecession_index')
