import numpy as np
import ray
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt, before_sleep_log

from class_data.data import Data
from class_generate.gen_emb import GenEmb
from utils.system import *

class GptTopic:
    def __init__(self,
                 query=None,
                 index=None,
                 score_col=None,
                 article_col=None,
                 topic_col=None,
                 top_n=None,
                 batch_size=None,
                 output=None
                 ):

        '''
        query (dict): Query used to generate
        index (pd.DataFrame): Generated index from either GenEmb, GenTfidf, GenCount
        score_col (str): Index score column name
        article_col (str): Index article column name
        topic_col (str): Index topic column name to be created
        top_n (float): Top n% scores from index
        batch_size (int): Batch size for parallelization
        output (str): Output name
        '''

        self.query = query
        self.index = index
        self.score_col = score_col
        self.article_col = article_col
        self.topic_col = topic_col
        self.top_n = top_n
        self.batch_size = batch_size
        self.output = output

    # Get top n% articles
    def get_top_n(self):
        sorted_index = self.index.sort_values(by=self.score_col, ascending=False)
        n_rows = int(len(sorted_index) * (self.top_n / 100))
        top_n_rows = sorted_index.head(n_rows)
        top_n_rows = top_n_rows.sort_index()
        top_n_rows.index = top_n_rows.index.to_pydatetime()
        return top_n_rows

    @staticmethod
    @ray.remote
    def gpt_topic(query, article_text):
        # Get first 200 words
        words = article_text.split()
        article_text = words[:100]
        article_text = ' '.join(article_text)

        api_key = json.load(open(get_config() / 'api.json'))['openai_api_key']
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": f"Identify and list the key subtopics within the field of {query} from this article using 1-3 words per subtopic: {article_text}. "
                                            f"If the article is not related to the field of {query}, output [null] as the subtopic list. "
                                            "Format the response as a JSON object: {\"subtopic\": [\"subtopic_1\", \"subtopic_2\", \"subtopic_3\", …]}. If unrelated, use {\"subtopic\": [null]}. "}
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
        subtopics = summary['subtopic']
        return subtopics

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), before_sleep=before_sleep_log(logger, logging.INFO))
    def retry_gpt_topic(self, query, article_text):
        # Call the remote function from within the retry logic
        return self.gpt_topic.remote(query, article_text)

    def get_gpt_topic(self, data):
        ray.init(num_cpus=16, ignore_reinit_error=True)
        num_batches = np.ceil(len(data) / self.batch_size)
        all_summary = []
        for i in tqdm(range(int(num_batches)), desc='Processing batches'):
            start_index = i * self.batch_size
            end_index = min(start_index + self.batch_size, len(data))
            batch = data[self.article_col][start_index:end_index]

            # Start asynchronous tasks for the batch
            futures = [self.retry_gpt_topic(self.query, text) for text in batch]
            batch_summaries = ray.get(futures)

            # Update lists
            all_summary.extend(batch_summaries)
            time.sleep(1)

        data[self.topic_col] = all_summary
        ray.shutdown()
        return data

    # Get topic overtime
    def get_topic(self):
        # Get top n rows
        index_top_n = self.get_top_n()

        # Fit topic model
        print("-"*60)
        print("Summarizing Articles...")
        topic = self.get_gpt_topic(index_top_n)
        topic.to_csv(f'../../plot/{self.output}/topic.csv', index=False)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------GET TOPICS--------------------------------------------------------------------
if __name__ == "__main__":
    # # Load openai embeddings
    # wsj_openai = Data(folder_path=get_format_data() / 'openai', file_pattern='wsj_emb_openai_*')
    # wsj_openai = wsj_openai.concat_files(10)
    #
    # # Load articles
    # wsj_art = Data(folder_path=get_format_data() / 'token', file_pattern='wsj_tokens_*')
    # wsj_art = wsj_art.concat_files(1)
    #
    # # Equal data
    # wsj_openai = wsj_openai.head(10000)
    # wsj_art = wsj_art.head(10000)

    # Load openai embeddings
    wsj_openai = Data(folder_path=get_format_data() / 'openai', file_pattern='wsj_emb_openai_*')
    wsj_openai = wsj_openai.concat_files()

    # Load articles
    wsj_art = Data(folder_path=get_format_data() / 'token', file_pattern='wsj_tokens_*')
    wsj_art = wsj_art.concat_files()

    # Params
    type = 'embedding'
    vector_column = 'ada_embedding'
    interval = 'M'
    threshold = 0.77

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-" * 120)
    query = 'Generate an index with label Artificial Intelligence from January 1st, 1984, to December 31st, 2021.'
    generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    index = generate.generate_emb()
    topic = GptTopic(query='Artificial Intelligence', index=index, score_col='score', article_col='body_txt', topic_col='topic', top_n=100, batch_size=16, output="ai_index")
    topic.get_topic()