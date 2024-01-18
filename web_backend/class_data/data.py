import glob
import pandas as pd
import re

from pathlib import Path

class Data:
    def __init__(self,
                 folder_path: Path = None,
                 file_pattern: str = None,
                 data: pd.DataFrame = None,
                 name: str = None):

        '''
        folder_path (Path): Path to folder where data is stored
        file_pattern (str): File pattern to identify files to concat
        data (pd.DataFrame): Pandas Dataframe to format
        name (str): Name of dataset to format
        '''

        self.folder_path = folder_path
        self.file_pattern = file_pattern
        self.data = data
        self.name = name

    # Concat all embedding files
    def concat_files(self):
        full_pattern = f'{self.folder_path}/{self.file_pattern}'
        file_list = glob.glob(full_pattern)
        number_extraction_pattern = self.file_pattern.replace('*', r'(\d+)')
        file_list.sort(key=lambda x: int(re.search(number_extraction_pattern, x).group(1)))
        df_list = [pd.read_parquet(file) for file in file_list]
        self.data = pd.concat(df_list, axis=0)
        return self.data

    # Format Dependent Variables
    def format_dep(self):
        if self.name == 'daily_us_news_index':
            # Set Date Column
            self.data['date'] = pd.to_datetime(self.data[['year', 'month', 'day']])
            self.data = self.data.set_index('date')
            self.data = self.data[['daily_policy_index']]
            # Rename column
            self.data.columns = ['daily_pol']
            return self.data
        elif self.name == 'categorical_epu_data':
            # Rename columns
            column_names = ['date', 'epu_month', 'mon_pol', 'fisc_pol', 'tax', 'gov_spend', 'health_care',
                            'nat_sec', 'ent_prog', 'reg', 'fin_reg', 'trade_pol', 'debt']
            self.data.columns = column_names
            # Remove Last Row
            self.data = self.data.iloc[:-1]
            # Convert date column to pd.datetime
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.set_index('date')
            self.data = self.data.apply(lambda group: group.resample('M').ffill())
            return self.data
        elif self.name == 'topic_attention':
            # Set index
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.set_index('date')
            # Rename Columns
            self.data.columns = ['topic_' + col.replace(' ', '_').replace('/', '_').replace('&', '_').lower() for col in self.data.columns]
            return self.data
        elif self.name == 'biodiversity_index':
            self.data = self.data[['month', 'attention']]
            self.data.columns = ['date', 'bio']
            self.data = self.data.dropna()
            self.data['date'] = self.data['date'].apply(lambda x: f"{x[:4]}-{x[5:].zfill(2)}-01")
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.set_index('date')
            self.data = self.data.apply(lambda group: group.resample('M').ffill())
            return self.data

    # Format Embedding
    def format_emb(self):
        if self.name == 'nyt' or self.name == 'wsj':
            # Set index to date
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.set_index('date')
            # Remove duplicate indices (keep the most recent date)
            self.data = self.data.loc[~self.data.index.duplicated(keep='last')]
            # Embeddings Columns
            data_emb_col = self.data.filter(regex='^c').columns
            # Calculate the average of the embeddings
            self.data[data_emb_col] = self.data[data_emb_col].div(self.data['tcount'], axis=0)
            # Rename embedding columns from 'c' to name
            rename_dict = {col: self.name + col[1:] for col in self.data.columns if col.startswith('c')}
            self.data.rename(columns=rename_dict, inplace=True)
            return self.data
        elif self.name == 'cc':
            # Set index to (permno)
            self.data = self.data.reset_index(level='fid', drop=True).reset_index(level='date')
            # Set date column to pd.datetime
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data['date'] = self.data['date'].dt.strftime('%Y-%m-%d')
            self.data['date'] = pd.to_datetime(self.data['date'])
            # Set index to (permno, date)
            self.data = self.data.reset_index().set_index(['permno', 'date']).sort_index(level=['permno', 'date'])
            # Remove duplicate indices (keep the most recent date)
            self.data = self.data.loc[~self.data.index.duplicated(keep='last')]
            return self.data

    # Format Article
    def format_article(self):
        if self.name == 'wsj_article':
            self.data['date'] = pd.to_datetime(self.data['display_date'])
            self.data['date'] = self.data['date'].dt.strftime('%Y-%m-%d')
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.set_index('date')
            self.data = self.data.drop(['display_date', 'doc_id'], axis=1)
            return self.data