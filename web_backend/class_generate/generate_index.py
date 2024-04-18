import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

class GenerateIndex:
    def __init__(self,
                 data=None,
                 p_val=None):

        '''
        data (dict): score and metadata for index
        p_val (str): p-value used for cosine similarity threshold
        '''

        self.data = data
        self.p_val = p_val

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


