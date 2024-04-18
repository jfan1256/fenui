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
                 expand=None,
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
        expand(bool): Expand query or not
        info (bool): Extract info or not (i.e., info=info if query='Systemic Financial Distress')
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
        self.expand = expand
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

    # Standardize
    @staticmethod
    def _standardize(index):
        # Min-max scale all indexes to 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        columns_to_scale = ['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']
        # Apply scaling for each column individually
        for column in columns_to_scale:
            data_to_scale = index[column].values.reshape(-1, 1)
            index[column] = scaler.fit_transform(data_to_scale)
        return index

    # Compare official index with generated index
    @staticmethod
    def _compare_index(index, interval, file_path):
        # Read in official data
        official_index = pd.read_parquet(get_format_data() / file_path)
        official_index.columns = ['official']
        # Resample official index to monthly or not
        if not official_index.index.freqstr == interval:
            official_index = official_index.resample(interval).mean()
        # Join official index to generated index
        index = index.join(official_index).dropna()
        # Min-max scale all indexes to 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        columns_to_scale = ['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score', 'official']
        # Apply scaling for each column individually
        for column in columns_to_scale:
            data_to_scale = index[column].values.reshape(-1, 1)
            index[column] = scaler.fit_transform(data_to_scale)
        # Calculate pearson correlation
        print("-" * 60)
        pearson_corr = index['relu_score'].corr(index['official'], method='pearson')
        print(f"Pearson Correlation with EPU: {pearson_corr}")
        return index, pearson_corr

    # Join official index with generated index into one dataframe
    @staticmethod
    def _join_index(index, interval, file_path):
        # Read in official data
        official_index = pd.read_parquet(get_format_data() / file_path)
        official_index.columns = ['official']
        # Resample official index to monthly or not
        if not official_index.index.freqstr == interval:
            official_index = official_index.resample(interval).mean()
        # Join official index to generated index
        index = index.join(official_index)
        # Min-max scale all indexes to 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        columns_to_scale = ['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score', 'official']
        # Apply scaling for each column individually
        for column in columns_to_scale:
            data_to_scale = index[column].values.reshape(-1, 1)
            index[column] = scaler.fit_transform(data_to_scale)
        return index

    # Save index, query, etc. to folder
    @staticmethod
    def save(query, expanded_query, p_val, threshold, pearson, index_paper, index_research, index_name_paper, index_name_research, output):
        # Make Dir
        plot_dir = f'../../view_attention/{output}'
        os.makedirs(plot_dir, exist_ok=True)

        # Save prompt and label
        with open(f'../../view_attention/{output}/{output}.txt', 'w') as file:
            file.write(f"Query: {query}\n\n")
            file.write(f"Expanded Query: {expanded_query}\n\n")
            file.write(f"P-Value: {p_val}\n\n")
            file.write(f"Threshold: {threshold}\n\n")
            file.write(f"Pearson Correlation: {pearson}\n\n")

        # Get plot
        plt.figure(figsize=(10, 5))
        colors = ['blue', 'red', 'orange', 'purple', 'cyan', 'green', 'magenta', 'yellow', 'black', 'grey']

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
        plt.savefig(f'../../view_attention/{output}/{output}.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()

        # Get plot
        plt.figure(figsize=(10, 5))
        colors = ['blue', 'red', 'orange', 'purple', 'cyan', 'green', 'magenta', 'yellow', 'black', 'grey']

        # Adjust thickness of axis
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.3)

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
        plt.savefig(f'../../view_attention/{output}/compare.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        return None

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
        if self.expand:
            self._expand_query()
        else:
            self.query['expanded_query'] = self.query['query']
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

        # Aggregate embeddings to a daily timeframe
        daily_aggregated = self.vector_data.groupby(self.vector_data.index.date)[self.vector_column].agg(lambda x: np.mean(np.stack(x), axis=0))
        vector_agg_matrix = np.stack(daily_aggregated.values)

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------COMPUTE COS SIM------------------------------------------------------------------
        print("-" * 60 + "\nComputing cosine similarity with label embedding...")
        cos_sim = cosine_similarity(label_emb, vector_matrix)
        self.vector_data['score'] = cos_sim[0]

        cos_sim_agg = cosine_similarity(label_emb, vector_agg_matrix)
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

        # Add embedding agg daily
        norm_score['agg_norm_score'] = cos_sim_agg[0]
        
        # Apply relu to daily timeframe
        scores = np.array(norm_score['norm_score'])
        percentile = 100 * (1 - 0.25)
        threshold = np.percentile(scores, percentile)
        relu_norm_score = np.maximum(0, norm_score['norm_score'] - threshold)
        relu_norm_score = relu_norm_score.to_frame('relu_norm_score')

        # Aggregate to monthly timeframe, yearly, or daily timeframe
        if self.interval == "M":
            # Retrieve top article per month
            monthly_art = art.groupby(pd.Grouper(freq='M')).apply(lambda x: x.nlargest(1, 'relu_score')).reset_index(level=0, drop=True)
            monthly_art.index = monthly_art.index.to_period('M').to_timestamp('M')

            # Join score and article
            relu_score = relu_score.resample('M').mean()
            norm_score = norm_score.resample('M').mean()
            relu_norm_score = relu_norm_score.resample('M').mean()
            relu_score = pd.concat([relu_score, relu_norm_score, norm_score, monthly_art[['headline', 'body_txt']]], axis=1)
        
        elif self.interval == "Y":
            # Retrieve top article per month
            yearly_art = art.groupby(pd.Grouper(freq='Y')).apply(lambda x: x.nlargest(1, 'relu_score')).reset_index(level=0, drop=True)
            yearly_art.index = yearly_art.index.to_period('Y').to_timestamp('Y')

            # Join score and article
            relu_score = relu_score.resample('Y').mean()
            norm_score = norm_score.resample('Y').mean()
            relu_norm_score = relu_norm_score.resample('Y').mean()
            relu_score = pd.concat([relu_score, relu_norm_score, norm_score, yearly_art[['headline', 'body_txt']]], axis=1)

        elif self.interval == "D":
            daily_art = art.groupby(art.index.date).first()
            daily_art.index = pd.to_datetime(daily_art.index)
            # Join score and article
            relu_score = pd.concat([relu_score, relu_norm_score, norm_score, daily_art[['headline', 'body_txt']]], axis=1)

        return relu_score, self.query['expanded_query'], threshold


# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------GET PLOTS---------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------PARAMS-----------------------------------------------------------------------
    # Params
    type = 'cc'
    expand = True
    info = False
    vector_column = 'ada_embedding'
    interval = 'Y'
    p_val = 0.01
    
    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------WSJ-----------------------------------------------------------------------
    if type == 'wsj':
        # Load openai embeddings
        vector_data = Data(folder_path=get_format_data() / 'openai', file_pattern='wsj_emb_textemb3small_*')
        vector_data = vector_data.concat_files()
    
        # Load articles
        article_data = Data(folder_path=get_format_data() / 'token', file_pattern='wsj_tokens_*')
        article_data = article_data.concat_files()

        # Set limit for number of articles per date
        limit = 30
        count = vector_data.groupby(vector_data.index)[vector_data.columns[0]].count()
        valid_date = count >= limit
        vector_data = vector_data[vector_data.index.isin(count[valid_date].index)]
        count = article_data.groupby(article_data.index)[article_data.columns[0]].count()
        valid_date = count >= limit
        article_data = article_data[article_data.index.isin(count[valid_date].index)]

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------CC-----------------------------------------------------------------------
    elif type == 'cc':
        # Load CC openai embeddings
        vector_data = Data(folder_path=get_format_data() / 'openai', file_pattern='cc_emb_textemb3small_*')
        vector_data = vector_data.concat_files()
    
        # Load CC articles
        article_data = Data(folder_path=get_format_data() / 'token', file_pattern='cc_tokens_*')
        article_data = article_data.concat_files()
    
        # Daily Multiple CC Metadata
        mdata = Data(folder_path=get_data() / 'cc_multiple', file_pattern='*_mdata.pq')
        mdata = mdata.concat_files()
    
        # Create date index
        mdata['date'] = pd.to_datetime(mdata['startDate'], format='%d-%b-%y %I:%M%p %Z')
        mdata['date'] = mdata['date'].dt.date
        mdata['date'] = pd.to_datetime(mdata['date'])
        mdata = mdata.set_index('fid')
    
        # Set index for CC embeddings
        vector_data.index = article_data['fid']
        vector_data = vector_data.join(mdata)
        vector_data = vector_data.reset_index().set_index('date').sort_index()
        vector_data = vector_data[['ada_embedding']]
        vector_data = vector_data.loc[~vector_data.ada_embedding.isnull()]
    
        # Set index for CC articles
        article_data = article_data.set_index('fid')
        article_data = article_data.join(mdata)
        article_data = article_data.rename(columns={'Headline': 'headline'})
        article_data = article_data.reset_index().set_index('date').sort_index()
        article_data = article_data.loc[~((article_data.index == '2006-10-18') & (article_data.fid == '1391246') & (article_data.content_type == 'Presentation'))]
        article_data = article_data[['headline', 'body_txt']]

    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-"*120)
    # query = 'ESG'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index = generate._join_index(index=index, interval=interval, file_path='esg_google_trend.parquet.brotli')
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['ESG', 'ESG (Google Trend)'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_esg_index')
    #
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-"*120)
    query = 'US Economic Policy Uncertainty'
    generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    index, expanded_query, threshold = generate.generate_emb()
    index, pearson_corr = generate._compare_index(index=index, interval=interval, file_path='epu.parquet.brotli')
    generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=pearson_corr, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['US EPU', 'US EPU (Baker et al.)'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_usepu_index')

    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-"*120)
    # query = 'Inflation'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index, pearson_corr = generate._compare_index(index=index, interval=interval, file_path='ir.parquet.brotli')
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=pearson_corr, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Inflation', '5-Year Breakeven Inflation Rate (FRED)'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_inflation_index')

    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Systemic Financial Stress'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index, pearson_corr = generate._compare_index(index=index, interval=interval, file_path='fsi.parquet.brotli')
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=pearson_corr, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Financial Stress', 'Financial Stress (Baker et al.)'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_fsi_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-"*120)
    # query = 'Economic Recession'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index, pearson_corr = generate._compare_index(index=index, interval=interval, file_path='recession.parquet.brotli')
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=pearson_corr, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Economic Recession', 'Economic Recession (Bybee et al.)'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_economicrecession_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Market Crash'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index = generate._standardize(index)
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Market Crash'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_marketcrash_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Stock Market Bubble'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index = generate._standardize(index)
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Stock Market Bubble'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_stockmarketbubble_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-"*120)
    # query = 'US-China Trade War'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index = generate._standardize(index)
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['US-China Trade War'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_uschinatradewar_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-"*120)
    # query = 'Artificial Intelligence'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index = generate._standardize(index)
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Artificial Intelligence'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_ai_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-"*120)
    # query = 'Blockchain'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index = generate._standardize(index)
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Blockchain'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_blockchain_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-"*120)
    # query = 'COVID-19'
    # generate = GenEmb(query=query, expand=False, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index = generate._standardize(index)
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['COVID-19'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_covid19_index')

    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-"*120)
    # query = "(OPERATOR INSTRUCTIONS) Jay Turner BMO Nesbitt Burns. Just a couple questions. Now the $450 million CapEx for the 16,000 tons of nickel, does this include anything at Fort Saskatchewan? Or how are you planning on refining those? This is Jowdat, Jay. The refinery will be expanded; but also the technology at the refinery will be changed. So the number includes both the expansion and the technology conversion to a (technical difficulty) extraction based system, which should substantially lower the energy requirements at Fort Saskatchewan. Okay, that was one thing. The Santa Cruz wells, are they going to be onshore? Are they going to be drilled from onshore? They will both be drilled from onshore, the same pad that the first well was drilled from. I see, okay. I know that you're cut (ph) off by confidentiality agreements on the Stelco stuff. But have you essential walked away from that whole process? Our bid's termination date was last Tuesday, and the bid expired pursuant to its termination provisions. So that is the end of our involvement, and at this stage we do not foresee any reason to look at it again. I see. So how does this impact now your strategy in Ontario for getting into the downstream energy business? Well, the strategy is actually not just in Ontario. It is basically in Canada generally. Our Bow City project is well on track. This year we're spending about $5 million of capital. We have already filed for the public review process to enable us to submit the application to proceed with that project. I think I may have missed something. Is this the old Brooks power project? It is now called the (multiple speakers). Bow City, yes, okay. I was wondering about (multiple speakers). If you see the newspapers and you see Bow City, that is the old Brooks project. For some reason I guess the project team decided to change the name. I don't know the reasons; don't ask me for those. We must have had somebody in our team from Bow City I suppose. \n But that is well on track. We on that project are proceeding together with Ontario Teachers. That right now, given the announcements the government has made vis-a-vis the grid, in our view, the next plant in Alberta. Just going back, I must have missed a bit on this. I was not aware of a decision that the Alberta government had made on the grid. Are they talking about aligning it more north-south now? Sorry, more east-west? There is a huge investment required to upgrade the grid between Edmonton and Calgary further. So they announced an upgrade to the facility; but they also I think at the same time announced that they were not looking to investing further billions of dollars in basically expanding it further. It doesn't make economic sense to do so. As a consequence now, a southern (ph) generation, which is where Bow City is located, becomes very attractive. It seems if my memory serves me that it was the east-west hook-up of that project that was a something block (multiple speakers). No, it was basically -- no, east-west is only for export to Saskatchewan or British Columbia. North-south was a basic corridor because the substantial capacity up in the north can generally be expanded cheaper than a greenfield capacity. So that is why the Genesee expansion that is now complete and running came ahead of Bow City, because it was cheaper to expand that plant before you could build a new greenfield plant in the south. But now the area around Edmonton (indiscernible) principally is tapped (ph) out. But what is the government proposing to do to help Brooks, specifically? I don't think we need any real help per se. The economics for the project stand on their own. We're not looking and we are not seeking at this stage any incentives from the government. And is this still going to be conceptually still an 800 MW facility that is being built, 2 by 400? I think you really would not be changing the stack size from a 450 unit. The system is capable of handling that. So the issue would be one unit or two units, whether it is 450 or it is two times that. Okay; and what kind of timeline for making a decision on that? I think the way the permitting goes, I am told it takes about six months to a year to get this through the energy board requirements. So we are in the first six-month phase right now. Okay, great. Thank you very much. (technical difficulty) Steve Bonnyman, CIBC World Markets. Following up on Jay's question, of the 450 million capital, that obviously provides for the expansion at Moa, the expansion at Fort Saskatchewan, and the technological shift. But the Cubans are contributing 25 years of reserves to this. What is the quid pro quo on that? They are contributing many things. They are contributing reserves and they are paying for half the capital. So you got a great deal, Jowdat. Is there no quid pro quo on the reserves? Remember, Steve, that the government takes the lion's share of the income. Not only do they get to keep (indiscernible) of the dividends but they also get taxes and royalties, which in the Cuban system start off when the capital cost is repaid, so maybe 7, 8 years from now on the expanded tonnage. So the government's, I think, return is based upon the jobs, the strategic importance of selling more nickel and cobalt, as well as future dividends and income. Fair enough. In terms of the expansion itself I think you said 30 months for the build. How long would the ramp up be? We do not anticipate that ramp up is going to be a big issue here. Remember that we are not doing a new construction. If anything we are adding more cranes in Moa. More tecnas (ph), more mining equipment, more leaching and precipitation capacity, and further refining capacity in Fort Saskatchewan. So I don't think there is going to be anything more than a six-month ramp up if things go well. That leads to my next question, because 450 million in capital is a pretty high number for what your proposing for a brownfield expansion. I assume that has to do at the technology conversion. Yes. I think if you want to use ballpark numbers on a (inaudible) basis, approximately $10 is in respect of the capacity part of it; and then the $2 and 70-odd cents on top of that is in respect of the technology change, which is hard to resist because the return is rather significant in going to an acid-based system as opposed to an ammonia-based system. Plus it is also proven technology, so there is no reason not to do it. Fair enough. But the plant -- what will you have to be pulling out of Fort Saskatchewan? Obviously the autoclaves if you are not using --. No, everything stays the way it is. There's -- You could use the same trains. Just what you use to leach it. Fair enough. So you are just changing (multiple speakers). How many shares to date have you been able to repurchase under the offer that is outstanding? None so far; the offer is still outstanding. I expected as much given where the share price had been. I just wanted to clarify that. We have no idea who will tender, who will not. I think the offer expires next week. Friday. At $10.68 nobody is going to tender. The last question was on the timing of the coal production expansions. You had indicated the production really would tick in towards the third quarter. Will we actually see any of that flow into sales next year? Or is the bulk of that impact going to be in the '06 year? It should actually flow into sales this year as well. We are projecting total volumes from the mine of 2.7 million tons this year which is going to be higher than the volumes this year by a fair amount. Then the full impact, the rest of the impact will be in '06. So the net impact in coal sales '04 over '05 you would expect to be what level? 2.7 compared to -- I am going to look at Guy. Where were we at? I think it's (indiscernible) 5% -- from a total production perspective it is about 5% increase when you add the Genesee expansion and Coal Valley together. We're projecting 40 million tons for this year compared to 38 last year. The Coal Valley part is 2.7, I think compared to just about 2. That's great, thank you very much. (OPERATOR INSTRUCTIONS) Anoop Prihar, GMP Securities. Just a couple of quick questions. What guidance would you provide for your consolidated tax rate for '05? I would say, Anoop, around 35% is probably a good number to use. Is there a split you can offer on cash versus deferred? I won't give you that, no. Just with respect to the accounting changes that you implemented in the fourth quarter, relative to the way that we reported the earnings in the third quarter, did these changes result in any meaningful change in earnings at all? I will let Susan answer that. I think the biggest one would be obviously the impact on our convertible debenture accounting. But let me let Susan give you the details. One element -- hi, Anoop -- that I will add before Susan picks up is that increasingly as we go into '05 and '06 I think it will be important to focus on two elements of earning. One is clearly the earnings as we are going to be reporting them. The other is the earnings from each of the growth that is currently underway, in metals and oil and coal and power. In each one of our principal segments there is committed growth which is a fairly substantial component, we hope certainly, of feature earnings. Susan? The changes with respect to the accounting changes, one was we adopted the VIE standard; as well as the second one was we adopted a standard with respect to the convertible debentures in the disclosure presentation. That is disclosed in note 2 to our interim consolidated financial statements. \n The cumulative effect of the change to the third quarter of '04 with respect to the VIE standard was an increase in EPS basic of $0.02, and that was just up to the third quarter. With respect to the convertible debenture change, the impact to third quarter was a decrease in EPS basic of $0.04; and that relates mostly to accretion charges on the debentures. Just so I understand then, relative to the number that we reported in Q4 with the accounting changes, if we had reported Q4 under the conventional or under the way we did it in the third quarter, would there have been a meaningful change in earnings per share? Only about $0.02 or $0.03. The accretion expense would not be there. It is mostly the accretion. That was $0.02, sorry? In the fourth quarter? In the fourth quarter it would have been $0.02 to $0.03. $0.02 to $0.3 basic. Thank you very much. Gary Chapman, Guardian Capital. I missed the first few minutes and perhaps you discussed this. But the nickel expansion, what IRR or return on investment are you using? And what nickel and cobalt prices go with that? I think it is better if you pick the nickel and cobalt prices. The thing that we are focused on is that for us this is a marginal expansion. So this will be -- all of the production will be going at marginal cost. We are earning (ph) around a buck, give or take a little, for our cash cost. The marginal cost will be substantially below that. So that is what we are focused on, the incremental marginal costs being substantially below the $1.10, $1.20 numbers that we are currently doing for cash costs. It makes sense perhaps, but in terms of the total capital put in the project you must have done an ROI. Would you have used long-term nickel of $3, $4, to make it just -- ? We like using $3.25; and $7.50 for cobalt. If I recall at one point earlier on you sort of thought about the expansion and that it would be brownfield, what would be relatively inexpensive to other projects. Then I think there's a time you're concerned about the cost of inputs rising and maybe it wouldn't be such a good idea. What has happened that makes it a good idea again? The basic issue ultimately was always return. Remember we own the business 50-50 with the government. In all previous iterations we were following the usual mode of investment in Cuba, where we provide all the capital and then we take all the cash flow before we are repaid; and then everybody shares 50-50. \n That is a very expensive exercise. I think the sea change here that happened was the government decided, on the strength of the economics, to basically fund their share on their own. That changes everything. So with that gone, the returns basically become irrelevant. Because no matter what prices you put in you're way above your IRR requirements. That makes a lot of sense. Thanks very much. There are no further questions at this time. Please continue. Ladies and gentlemen, thank you very much for listening; and please visit our webpage at www.Sherritt.com to access replay information. Thank you very much, Connie, and thank you all for your participation. Ladies and gentlemen, this concludes the conference call for today. Thank you for participating. Please disconnect your lines."
    # generate = GenEmb(query=query, expand=False, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index = generate._standardize(index)
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['COVID-19'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_tesla_index')

    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-"*120)
    # language = [('English', 'Inflation'), ('Chinese', '通货膨胀'), ('Russian', 'инфляция'), ('Spanish', 'inflación'), ('French', "d'inflation"), ('Arabic', 'زِيادة في الأَسْعار')]
    # for i, (lan, query) in enumerate(language):
    #     generate = GenEmb(query=query, expand=False, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    #     index, expanded_query, threshold = generate.generate_emb()
    #     if lan == 'English':
    #         keep = index
    #         continue
    #     print(f"US and {lan} Correlation: {keep['relu_score'].corr(index['relu_score'])}")
