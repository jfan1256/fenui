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
                 info=None,
                 type=None,
                 vector_data=None,
                 vector_column=None,
                 preprocess_data=None,
                 preprocess_column=None,
                 article_data=None,
                 interval=None,
                 p_val=None,
                 ):

        '''
        query (str): User input (should contain a label, start date and end date)
        info (bool): Extract info or not (i.e., info=False if query='Systemic Financial Distress')
        type (str): Method to generate an index (either 'embedding', 'tfidf', 'count')
        vector_data (pd.DataFrame): Pandas dataframe that stores the article vectors
        vector_column (str): Column name for the vector column
        preprocess_data (pd.DataFrame): Pandas dataframe that stores the preprocessed articles
        preprocess_column (str): Column name for the preprocess column
        article_data (pd.DataFrame): Pandas dataframe that stores the article headline and body text
        interval (str): Date interval for generated index (either 'D' or 'M')
        p_val (int): Significance Level for Threshold parameter for transformation
        '''

        self.query = query
        self.info = info
        self.type = type
        self.vector_data = vector_data
        self.vector_column = vector_column
        self.preprocess_data = preprocess_data
        self.preprocess_column = preprocess_column
        self.article_data = article_data
        self.interval = interval
        self.p_val = p_val

        api_key = json.load(open(get_config() / 'api.json'))['openai_api_key']
        self.client = OpenAI(api_key=api_key)

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

    # Extract info
    def _extract_info(self):
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

    # Expand Query
    def _expand_query(self):
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
        self.query['expanded_query'] = paragraph

    # Get query's openai embedding
    def _get_emb(self):
        text = self.query['expanded_query'].replace("\n", " ")
        return self.client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

    # Compare official index with generated index
    @staticmethod
    def _compare_index(index, file_path):
        # Read in official data
        official_index = pd.read_parquet(get_format_data() / file_path)
        official_index.columns = ['official']
        # Resample official index to monthly or not
        if not official_index.index.freqstr == 'M':
            official_index = official_index.resample('M').mean()
        # Join official index to generated index
        index = index.join(official_index).dropna()
        # Min-max scale all indexes to 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_to_scale = index[['relu_score', 'norm_score', 'official']]
        scaled_data = scaler.fit_transform(data_to_scale)
        index[['relu_score', 'norm_score', 'official']] = scaled_data
        # Calculate pearson correlation
        print("-" * 60)
        pearson_corr = index['relu_score'].corr(index['official'], method='pearson')
        print(f"Pearson Correlation with EPU: {pearson_corr}")
        return index, pearson_corr

    # Join official index with generated index into one dataframe
    @staticmethod
    def _join_index(index, file_path):
        # Read in official data
        official_index = pd.read_parquet(get_format_data() / file_path)
        official_index.columns = ['official']
        # Resample official index to monthly or not
        if not official_index.index.freqstr == 'M':
            official_index = official_index.resample('M').mean()
        # Join official index to generated index
        index = index.join(official_index)
        # Min-max scale all indexes to 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_to_scale = index[['relu_score', 'norm_score', 'official']]
        scaled_data = scaler.fit_transform(data_to_scale)
        index[['relu_score', 'norm_score', 'official']] = scaled_data
        return index

    # Save index, query, etc. to folder
    @staticmethod
    def save(query, expanded_query, p_val, threshold, pearson, index_paper, index_research, index_name_paper, index_name_research, output):
        # Make Dir
        plot_dir = f'../../plot/{output}'
        os.makedirs(plot_dir, exist_ok=True)

        # Save prompt and label
        with open(f'../../plot/{output}/{output}.txt', 'w') as file:
            file.write(f"Query: {query}\n\n")
            file.write(f"Expanded Query: {expanded_query}\n\n")
            file.write(f"P-Value: {p_val}\n\n")
            file.write(f"Threshold: {threshold}\n\n")
            file.write(f"Pearson Correlation: {pearson}\n\n")

        # Get plot
        plt.figure(figsize=(10, 5))
        colors = ['blue', 'red', 'cyan', 'orange', 'green', 'purple', 'magenta', 'yellow', 'black', 'grey']

        # Adjust thickness of axis
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.3)

        # Plot each line for index_paper (Plot to put in Paper)
        for i, (column, name) in enumerate(zip(index_paper.columns, index_name_paper)):
            plt.plot(index_paper.index, index_paper[column], label=f"{name} Index", color=colors[i % len(colors)])

        # Set up legend, x-axis, y-axis, and grid
        plt.legend(fontsize='medium', prop={'weight': 'semibold'}, loc='upper left')
        plt.ylabel("Attention Index", color='black', fontweight='semibold', fontsize=12, labelpad=15)
        plt.xlabel("", color='black', fontweight='semibold', fontsize=12, labelpad=15)
        plt.xticks(color='black', fontweight='semibold')
        plt.tick_params(axis='x', which='major', direction='out', length=5, width=1.3, colors='black', labelsize=10, pad=5)
        plt.tick_params(axis='y', which='both', left=False, labelleft=False, pad=10)
        plt.grid(False)

        # Save figure
        plt.savefig(f'../../plot/{output}/{output}.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()

        # Plot each line for index_research (Plot to see threshold index vs. no threshold index)
        for i, (column, name) in enumerate(zip(index_research.columns, index_name_research)):
            plt.plot(index_research.index, index_research[column], label=f"{name} Index", color=colors[i % len(colors)])

        # Set up legend, x-axis, y-axis, and grid
        plt.legend(fontsize='medium', prop={'weight': 'semibold'}, loc='upper left')
        plt.ylabel("Attention Index", color='black', fontweight='semibold', fontsize=12, labelpad=15)
        plt.xlabel("", color='black', fontweight='semibold', fontsize=12, labelpad=15)
        plt.xticks(color='black', fontweight='semibold')
        plt.tick_params(axis='x', which='major', direction='out', length=5, width=1.3, colors='black', labelsize=10, pad=5)
        plt.tick_params(axis='y', which='both', left=False, labelleft=False, pad=10)
        plt.grid(False)

        # Save figure
        plt.savefig(f'../../plot/{output}/compare.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()

    # Generate Embedding Index
    def generate_emb(self):
        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------EXTRACT INFO--------------------------------------------------------------------
        # Extract info (i.e., start_date, end_date, etc.)
        if self.info:
            self._extract_info()
        else:
            self.query = dict(query=self.query, start_date='1984-01-01', end_date='2021-12-31')

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------EXPAND QUERY--------------------------------------------------------------------
        # Expand query
        self._expand_query()
        print(f"Here is the query: \n{self.query}")

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------SET TIMEFRAME-------------------------------------------------------------------
        # Set timeframe
        if self.vector_data is not None and not self.vector_data.empty:
            self.vector_data = self.vector_data[(self.vector_data.index >= self.query["start_date"]) & (self.vector_data.index <= self.query["end_date"])]
        if self.preprocess_data is not None and not self.preprocess_data.empty:
            self.preprocess_data = self.preprocess_data[(self.preprocess_data.index >= self.query["start_date"]) & (self.preprocess_data.index <= self.query["end_date"])]
        if self.article_data is not None and not self.article_data.empty:
            self.article_data = self.article_data[(self.article_data.index >= self.query["start_date"]) & (self.article_data.index <= self.query["end_date"])]

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------GET QUERY VECTOR------------------------------------------------------------------
        print("-" * 60 + "\nGetting query vector...")
        label_emb = self._get_emb()
        label_emb = np.array(label_emb).reshape(1, -1)

        # Convert vector_data to a matrix
        vector_matrix = np.stack(self.vector_data[self.vector_column].values)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------COMPUTE COS SIM------------------------------------------------------------------
        print("-" * 60 + "\nComputing cosine similarity with label embedding...")
        cos_sim = cosine_similarity(label_emb, vector_matrix)
        self.vector_data['score'] = cos_sim[0]

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------APPLY TRANSFORMATION---------------------------------------------------------------
        # Get threshold p_val
        scores = np.array(self.vector_data['score'])
        percentile = 100 * (1 - self.p_val)
        threshold = np.percentile(scores, percentile)

        # Apply RELU transformation
        relu_score = np.maximum(0, self.vector_data['score'] - threshold)

        # No RELU transformation
        norm_score = self.vector_data['score']

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------AGGREGATE--------------------------------------------------------------------
        relu_score.index.names = ['date']
        norm_score.index.names = ['date']
        relu_score = relu_score.to_frame('relu_score')
        norm_score = norm_score.to_frame('norm_score')

        # Get articles
        art = pd.concat([relu_score, self.article_data[['headline', 'body_txt']]], axis=1)
        art = art.sort_values(by='relu_score', ascending=False)

        # Aggregate to daily timeframe
        relu_score = relu_score.groupby('date').mean()
        relu_score.columns = ['relu_score']
        norm_score = norm_score.groupby('date').mean()
        norm_score.columns = ['norm_score']

        # Aggregate to monthly timeframe or daily timeframe
        if self.interval == "M":
            # Retrieve top article per month
            monthly_art = art.groupby(pd.Grouper(freq='M')).apply(lambda x: x.nlargest(1, 'relu_score')).reset_index(level=0, drop=True)
            monthly_art.index = monthly_art.index.to_period('M').to_timestamp('M')

            # Join score and article
            relu_score = relu_score.resample('M').mean()
            norm_score = norm_score.resample('M').mean()
            relu_score = pd.concat([relu_score, norm_score, monthly_art[['headline', 'body_txt']]], axis=1)

        elif self.interval == "D":
            daily_art = art.groupby(art.index.date).first()
            daily_art.index = pd.to_datetime(daily_art.index)
            # Join score and article
            relu_score = pd.concat([relu_score, norm_score, daily_art[['headline', 'body_txt']]], axis=1)

        return relu_score, self.query['expanded_query'], threshold


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
    p_val = 0.01

    # # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # print("-"*120)
    # # query = 'Generate an index with label ESG from January 1st, 1984, to December 31st, 2021.'
    # # generate = GenEmb(query=query, info=True, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, p_val=p_val)
    # # index, expanded_query, threshold = generate.generate_emb()
    # # index = generate._join_index(index=index, file_path='esg_google_trend.parquet.brotli')
    # # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'norm_score']], index_name_paper=['ESG', 'ESG (Google Trend)'], index_name_research=['Transformed', 'Non-Transformed'], output='esg_index')

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-"*120)
    query = 'ESG'
    generate = GenEmb(query=query, info=False, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, p_val=p_val)
    index, expanded_query, threshold = generate.generate_emb()
    index = generate._join_index(index=index, file_path='esg_google_trend.parquet.brotli')
    generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'norm_score']], index_name_paper=['ESG', 'ESG (Google Trend)'], index_name_research=['Transformed', 'Non-Transformed'], output='esg_index')

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-"*120)
    query = 'US Economic Policy Uncertainty'
    generate = GenEmb(query=query, info=False, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, p_val=p_val)
    index, expanded_query, threshold = generate.generate_emb()
    index, pearson_corr = generate._compare_index(index=index, file_path='epu.parquet.brotli')
    generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=pearson_corr, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'norm_score']], index_name_paper=['US EPU', 'US EPU (Baker et al.)'], index_name_research=['Transformed', 'Non-Transformed'], output='usepu_index')

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-"*120)
    query = 'Inflation'
    generate = GenEmb(query=query, info=False, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, p_val=p_val)
    index, expanded_query, threshold = generate.generate_emb()
    index, pearson_corr = generate._compare_index(index=index, file_path='ir.parquet.brotli')
    generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=pearson_corr, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'norm_score']], index_name_paper=['Inflation', '5-Year Breakeven Inflation Rate (FRED)'], index_name_research=['Transformed', 'Non-Transformed'], output='inflation_index')

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-" * 120)
    query = 'Systemic Financial Stress'
    generate = GenEmb(query=query, info=False, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, p_val=p_val)
    index, expanded_query, threshold = generate.generate_emb()
    index, pearson_corr = generate._compare_index(index=index, file_path='fsi.parquet.brotli')
    generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=pearson_corr, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'norm_score']], index_name_paper=['Financial Stress', 'Financial Stress (Baker et al.)'], index_name_research=['Transformed', 'Non-Transformed'], output='fsi_index')

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-"*120)
    query = 'Economic Recession'
    generate = GenEmb(query=query, info=False, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, p_val=p_val)
    index, expanded_query, threshold = generate.generate_emb()
    index, pearson_corr = generate._compare_index(index=index, file_path='recession.parquet.brotli')
    generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=pearson_corr, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'norm_score']], index_name_paper=['Economic Recession', 'Economic Recession (Bybee et al.)'], index_name_research=['Transformed', 'Non-Transformed'], output='economicrecession_index')

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-" * 120)
    query = 'Market Crash'
    generate = GenEmb(query=query, info=False, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, p_val=p_val)
    index, expanded_query, threshold = generate.generate_emb()
    generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'norm_score']], index_name_paper=['US-China Trade War'], index_name_research=['Transformed', 'Non-Transformed'], output='uschinatradewar_index')

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-" * 120)
    query = 'Stock Market Bubble'
    generate = GenEmb(query=query, info=False, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, p_val=p_val)
    index, expanded_query, threshold = generate.generate_emb()
    generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'norm_score']], index_name_paper=['US-China Trade War'], index_name_research=['Transformed', 'Non-Transformed'], output='uschinatradewar_index')

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-"*120)
    query = 'US-China Trade War'
    generate = GenEmb(query=query, info=False, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, p_val=p_val)
    index, expanded_query, threshold = generate.generate_emb()
    generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'norm_score']], index_name_paper=['US-China Trade War'], index_name_research=['Transformed', 'Non-Transformed'], output='uschinatradewar_index')

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-"*120)
    query = 'Artificial Intelligence'
    generate = GenEmb(query=query, info=False, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, p_val=p_val)
    index, expanded_query, threshold = generate.generate_emb()
    generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'norm_score']], index_name_paper=['Artificial Intelligence'], index_name_research=['Transformed', 'Non-Transformed'], output='ai_index')

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-"*120)
    query = 'Blockchain'
    generate = GenEmb(query=query, info=False, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, p_val=p_val)
    index, expanded_query, threshold = generate.generate_emb()
    generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'norm_score']], index_name_paper=['Blockchain'], index_name_research=['Transformed', 'Non-Transformed'], output='blockchain_index')

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-"*120)
    query = 'COVID-19'
    generate = GenEmb(query=query, info=False, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, p_val=p_val)
    index, expanded_query, threshold = generate.generate_emb()
    generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'norm_score']], index_name_paper=['COVID-19'], index_name_research=['Transformed', 'Non-Transformed'], output='covid19_index')