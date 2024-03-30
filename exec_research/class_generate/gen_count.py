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

class GenCount:
    def __init__(self,
                 query=None,
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

        # Count Extraction
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": f"Here is the customer input: {self.query} "
                                            "Execute these steps: "
                                            "Step 1) Extract the label, start date, and end date from this piece of text. "
                                            "Step 2) If you cannot extract a label, start date, or end date, then store the them as 'none'. "
                                            "Step 3) Output this: {label: label, start_date: YYYY-MM-DD, end_date: YYYY-MM-DD}. "
                                            "Provide the output in JSON format."
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

    # Generate Count Index
    def generate_count(self):
        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------COMPUTE SCORE-------------------------------------------------------------------
        print("-" * 60 + "\nComputing score...")
        # Compute score
        pattern = rf'\b{self.query["label"].lower()}\b'
        self.preprocess_data['score'] = self.preprocess_data[self.preprocess_column].str.lower().str.count(pattern)
        score = self.preprocess_data[['score']]

        # --------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------AGGREGATE---------------------------------------------------------------------
        # Add article
        art = pd.concat([score, self.article_data[['headline', 'body_txt']]], axis=1)
        art = art.sort_values(by='score', ascending=False)

        # Aggregate to daily timeframe
        score = score.groupby('date').sum()
        score.columns = ['score']

        # Aggregate to monthly timeframe or daily timeframe
        if self.interval == "M":
            # Retrieve top article per month
            monthly_art = art.groupby(pd.Grouper(freq='M')).apply(lambda x: x.nlargest(1, 'score')).reset_index(level=0, drop=True)
            monthly_art.index = monthly_art.index.to_period('M').to_timestamp('M')

            # Join score and article
            score = score.resample('M').mean()
            score = pd.concat([score, monthly_art[['headline', 'body_txt']]], axis=1)

        elif self.interval == "D":
            daily_art = art.groupby(art.index.date).first()
            daily_art.index = pd.to_datetime(daily_art.index)
            # Join score and article
            score = pd.concat([score, daily_art[['headline', 'body_txt']]], axis=1)

        return score

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------GET PLOT---------------------------------------------------------------------
if __name__ == "__main__":
    # Load articles
    wsj_art = Data(folder_path=get_format_data() / 'token', file_pattern='wsj_tokens_*')
    wsj_art = wsj_art.concat_files()

    # Params
    type = 'count'
    preprocess_column = 'body_txt'
    interval = 'M'

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-"*120)
    query = 'Generate an index with label ESG from January 1st, 1984, to December 31st, 2021.'
    generate = GenCount(query=query, preprocess_data=wsj_art, preprocess_column=preprocess_column, article_data=wsj_art, interval=interval)
    index = generate.generate_count()
    generate.exec_plot(prompt=query, label=generate.query['label'], pearson=0, data=index[['score']], names=['ESG'], output='esgcount_index')
