import glob
import pandas as pd
import re

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm


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

    # Concat all embedding files (not parallelized)
    def concat_files_slow(self, num=None):
        full_pattern = f'{self.folder_path}/{self.file_pattern}'
        file_list = glob.glob(full_pattern)
        number_extraction_pattern = self.file_pattern.replace('*', r'(\d+)')
        file_list.sort(key=lambda x: int(re.search(number_extraction_pattern, x).group(1)))
        if num is not None:
            df_list = []
            for i, file in enumerate(tqdm(file_list[:num], desc="Loading Data")):
                df_list.append(pd.read_parquet(file))
        else:
            df_list = [pd.read_parquet(file) for file in tqdm(file_list, desc="Loading Data")]
        self.data = pd.concat(df_list, axis=0)
        return self.data

    # Read parquet file
    def read_parquet_file(self, file):
        return file, pd.read_parquet(file)

    # Concat all embedding files (parallelized)
    def concat_files(self, num=None):
        # Get file
        full_pattern = f'{self.folder_path}/{self.file_pattern}'
        file_list = glob.glob(full_pattern)
        number_extraction_pattern = self.file_pattern.replace('*', r'(\d+)')
        file_list.sort(key=lambda x: int(re.search(number_extraction_pattern, x).group(1)))

        # Adjust num to the length of file_list if it's None or greater than the list
        num = num if num is not None and num <= len(file_list) else len(file_list)

        # Read files in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.read_parquet_file, file) for file in file_list[:num]]
            df_dict = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading Data"):
                file, df = future.result()
                df_dict[file] = df

        # Ensure DataFrames are concatenated in the correct order
        ordered_dfs = [df_dict[file] for file in file_list[:num]]
        self.data = pd.concat(ordered_dfs, axis=0)
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
        elif self.name == 'recession_attention':
            self.data.date = pd.to_datetime(self.data.date).dt.to_period('M').dt.to_timestamp('M')
            self.data = self.data.set_index('date')
            self.data.columns = ['recession']
            return self.data
        elif self.name == 'epu_data':
            # Remove Last Row
            self.data = self.data.iloc[:-1]
            # Set index
            self.data['date'] = pd.to_datetime(self.data['Year'].astype(int).astype(str) + '-' + self.data['Month'].astype(int).astype(str) + '-01').dt.to_period("M").dt.to_timestamp("M")
            self.data = self.data.set_index('date')
            # Get correct column
            self.data = self.data[['News_Based_Policy_Uncert_Index']]
            self.data.columns = ['epu']
            return self.data
        elif self.name == 'categorical_epu_data':
            # Rename columns
            column_names = ['date', 'epu_month', 'mon_pol', 'fisc_pol', 'tax', 'gov_spend', 'health_care', 'nat_sec', 'ent_prog', 'reg', 'fin_reg', 'trade_pol', 'debt']
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
        elif self.name == 'ai_google_trend':
            self.data.columns = ['date', 'ai']
            self.data.date = pd.to_datetime(self.data.date).dt.to_period('M').dt.to_timestamp('M')
            self.data = self.data.set_index('date')
            return self.data
        elif self.name == 'esg_google_trend':
            self.data.columns = ['date', 'esg']
            self.data.date = pd.to_datetime(self.data.date).dt.to_period('M').dt.to_timestamp('M')
            self.data = self.data.set_index('date')
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