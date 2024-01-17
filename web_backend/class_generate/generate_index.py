import pandas as pd
import numpy as np

class GenerateIndex:
    def __init__(self,
                 query=None,
                 transform=None):

        '''
        query (ChromaDB Dict): Returned dictionary after a ChromaDB Query
        transform (str): Cosine similarity transformation
        '''

        self.query = query
        self.transform = transform

    # Aggregate daily
    @staticmethod
    def _agg_daily(data, labels, name):
        data_agg = data[labels].to_frame()
        data_agg.index.names = ['date']
        data_agg = data_agg.groupby('date').mean()
        data_agg.columns = [name]
        return data_agg

    # Generate uncertainty index
    def generate_index(self):
        # Prepare lists
        dates = []
        cos_sims = []
        headlines = []
        documents = []

        # Iterate through each date's results in the query response
        for daily_results in self.query:
            for result in daily_results.values():
                for article in result:
                    dates.append(article['date'])
                    cos_sims.append(article['score'])
                    headlines.append(article['headline'])
                    documents.append(article['document'])

        # Create pandas dataframe
        data = pd.DataFrame({
            'date': dates,
            'cos_sim': cos_sims,
            'headline': headlines,
            'document': documents
        })
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date').sort_index()

        # Apply transformation
        if self.transform == 'relu':
            data['transform_cos_sim'] = np.maximum(0, data['cos_sim'] - 0.75)
        elif self.transform == 'square relu':
            data['transform_cos_sim'] = np.maximum(0, data['cos_sim'] - 0.75) ** 2
        elif self.transform == 'sigmoid':
            data['transform_cos_sim'] = 1 / (1 + np.exp(-(0.5 * data['cos_sim'] + -1)))
        elif self.transform == 'arcsin':
            data['transform_cos_sim'] = np.arcsin(np.clip(data['cos_sim'], -1, 1))

        # Create daily uncertainty index
        gen_index = self._agg_daily(data=data, labels='transform_cos_sim', name='daily_cos_sim')

        # Create daily article index (largest cosine similarity)
        max_cos_sim = data.groupby('date')['transform_cos_sim'].transform('max')
        mask = data['transform_cos_sim'] == max_cos_sim
        gen_article = data[mask][['headline', 'document']]
        gen_article.columns = ['daily_headline', 'daily_document']

        # Combine daily uncertainty index and daily article index
        gen_combine = pd.concat([gen_index, gen_article], axis=1)
        gen_combine.index = gen_combine.index.strftime('%Y-%m-%d')

        return gen_index, gen_combine


