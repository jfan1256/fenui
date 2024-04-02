import pandas as pd
import numpy as np

class GenerateIndex:
    def __init__(self,
                 query=None,
                 p_vaL=None):

        '''
        query (str): Expanded query
        p_vaL (str): p-value used for cosine similarity threshold
        '''

        self.query = query
        self.p_val = p_vaL

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
        return gen_index, gen_combine


