import json
import numpy as np
import pandas as pd

from openai import OpenAI
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

from utils.system import get_config

class GenerateIndex:
    def __init__(self,
                 data=None,
                 query=None,
                 p_val=None):

        '''
        data (dict): score and metadata for index
        query (str): query
        p_val (str): p-value used for cosine similarity threshold
        '''

        self.data = data
        self.query = query
        self.p_val = p_val

        # Get API Key
        api = json.load(open(get_config() / 'api.json'))
        self.api_key = api['openai_api_key']

        # Define OpenAI model
        self.model = 'text-embedding-3-small'

    # Aggregate daily
    @staticmethod
    def _agg_daily(data, labels, name):
        data_agg = data[labels].to_frame()
        data_agg.index.names = ['date']
        data_agg = data_agg.groupby('date').mean()
        data_agg.columns = [name]
        return data_agg

    # Get OpenAI embedding
    def _get_openai_emb(self, text):
        client = OpenAI(api_key=self.api_key)
        embedding = client.embeddings.create(input=[text.replace("\n", " ")], model=self.model).data[0].embedding
        return embedding

    # Generate attention index for database
    def generate_index_db(self):
        # Create pandas dataframe
        data = pd.DataFrame(self.data)
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date').sort_index()

        # Apply transformation (p-value percentile threshold)
        scores = np.array(data['cos_sim'])
        percentile = 100 * (1 - self.p_val)
        threshold = np.percentile(scores, percentile)
        data['relu_score'] = np.maximum(0, data['cos_sim'] - threshold)

        # Create daily uncertainty index
        daily_index = self._agg_daily(data=data, labels='relu_score', name='relu_score')

        # Setup article index
        art = data[['relu_score', 'headline', 'document']]
        art = art.sort_values(by='relu_score', ascending=False)

        # Retrieve top article per month
        monthly_art = art.groupby(pd.Grouper(freq='M')).apply(lambda x: x.nlargest(1, 'relu_score')).reset_index(level=0, drop=True)
        monthly_art.index = monthly_art.index.to_period('M').to_timestamp('M')
        monthly_art = monthly_art[['headline', 'document']]

        # Join score and article
        gen_index = daily_index.resample('M').mean()
        gen_combine = pd.concat([gen_index, monthly_art], axis=1)
        gen_index.index = gen_index.index.strftime('%Y-%m-%d')
        gen_combine.index = gen_combine.index.strftime('%Y-%m-%d')

        # Min Max
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_to_scale = gen_index[['relu_score']]
        scaled_data = scaler.fit_transform(data_to_scale)
        gen_index[['relu_score']] = scaled_data

        # Rename column
        gen_index.columns = ['attention']
        return gen_index, gen_combine

    # # Generate attention index for parquet files
    # def generate_index_pq(self):
    #     # Get query embedding
    #     print("Get query embedding")
    #     query_emb = self._get_openai_emb(self.query)
    #     query_emb = np.array(query_emb).reshape(1, -1)
    #
    #     # Convert vector_data to a matrix
    #     vector_matrix = np.stack(self.data['ada_embedding'].values)
    #
    #     # Calculate score
    #     print("Calculate Score")
    #     score = cosine_similarity(query_emb, vector_matrix)[0]
    #
    #     # Create dataframe
    #     data = self.data[['headline', 'body_txt']]
    #     data['cos_sim'] = score
    #     data = data.rename(columns={'body_txt':'document'})
    #
    #     # Apply transformation (p-value percentile threshold)
    #     scores = np.array(data['cos_sim'])
    #     percentile = 100 * (1 - self.p_val)
    #     threshold = np.percentile(scores, percentile)
    #     data['relu_score'] = np.maximum(0, data['cos_sim'] - threshold)
    #
    #     # Create daily uncertainty index
    #     print("Aggregate Daily")
    #     daily_index = self._agg_daily(data=data, labels='relu_score', name='relu_score')
    #
    #     # Setup article index
    #     art = data[['relu_score', 'headline', 'document']]
    #     art = art.sort_values(by='relu_score', ascending=False)
    #
    #     # Retrieve top article per month
    #     print("Aggregate Monthly")
    #     monthly_art = art.groupby(pd.Grouper(freq='M')).apply(lambda x: x.nlargest(1, 'relu_score')).reset_index(level=0, drop=True)
    #     monthly_art.index = monthly_art.index.to_period('M').to_timestamp('M')
    #     monthly_art = monthly_art[['headline', 'document']]
    #
    #     # Join score and article
    #     gen_index = daily_index.resample('M').mean()
    #     gen_combine = pd.concat([gen_index, monthly_art], axis=1)
    #     gen_index.index = gen_index.index.strftime('%Y-%m-%d')
    #     gen_combine.index = gen_combine.index.strftime('%Y-%m-%d')
    #
    #     # Min Max
    #     print("Min-Max Scale")
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     data_to_scale = gen_index[['relu_score']]
    #     scaled_data = scaler.fit_transform(data_to_scale)
    #     gen_index[['relu_score']] = scaled_data
    #
    #     # Rename column
    #     gen_index.columns = ['attention']
    #     return gen_index, gen_combine
    def generate_index_pq(self):
        print("Get query embedding")
        query_emb = self._get_openai_emb(self.query).reshape(1, -1)

        print("Calculate Score")
        vector_matrix = np.stack(self.data['ada_embedding'].values)
        scores = cosine_similarity(query_emb, vector_matrix)[0]

        self.data['cos_sim'] = scores
        threshold = np.percentile(scores, 100 * (1 - self.p_val))
        self.data['relu_score'] = np.maximum(0, self.data['cos_sim'] - threshold)

        print("Aggregate Daily")
        daily_index = self._agg_daily(data=self.data, labels='relu_score', name='relu_score')

        print("Aggregate Monthly")
        monthly_art = self.data.nlargest(1, 'relu_score', keep='all')
        monthly_art.index = monthly_art.index.to_period('M').to_timestamp('M')
        monthly_art = monthly_art[['headline', 'body_txt']].rename(columns={'body_txt':'document'})

        print("Min-Max Scale")
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(daily_index.to_frame())
        daily_index = pd.DataFrame(scaled_data, index=daily_index.index, columns=['attention'])

        return daily_index, monthly_art

