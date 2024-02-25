from bertopic import BERTopic

from class_data.data import Data
from class_generate.gen_emb import GenEmb
from utils.system import get_format_data

class Topic:
    def __init__(self,
                 index=None,
                 score_col=None,
                 article_col=None,
                 top_n=None,
                 n_topic=None,
                 output=None
                 ):

        '''
        index (pd.DataFrame): Generated index from either GenEmb, GenTfidf, GenCount
        score_col (str): Index score column name
        article_col (str): Index article column name
        top_n (float): Top n% scores from index
        n_topic (int): Number of topics to extract
        output (str): Output name
        '''

        self.index = index
        self.score_col = score_col
        self.article_col = article_col
        self.top_n = top_n
        self.n_topic = n_topic
        self.output = output

    # Get top n% articles
    def get_top_n(self):
        sorted_index = self.index.sort_values(by=self.score_col, ascending=False)
        n_rows = int(len(sorted_index) * (self.top_n / 100))
        top_n_rows = sorted_index.head(n_rows)
        top_n_rows = top_n_rows.sort_index()
        top_n_rows.index = top_n_rows.index.to_pydatetime()
        return top_n_rows

    # Get topic overtime
    def get_topic(self):
        # Get top n rows
        index_top_n = self.get_top_n()
        articles = index_top_n[self.article_col].tolist()
        timestamps = index_top_n.index.tolist()

        # Fit topic model
        print("-"*60)
        print("Fitting Model...")
        topic_model = BERTopic(verbose=True)
        topics, probs = topic_model.fit_transform(articles)

        # Topics over time
        print("-"*60)
        print("Getting topics over time...")
        topics_over_time = topic_model.topics_over_time(articles, timestamps, global_tuning=False, evolution_tuning=True, nr_bins=self.n_topic)
        topics_over_time.to_csv(f'../../plot/{self.output}/topic.csv', index=False)

        # Visualize Topics
        print("-"*60)
        print("Visualizing...")
        fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=self.n_topic)
        fig.write_html(f'../../plot/{self.output}/topic.html')

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
    interval = 'D'
    threshold = 0.77

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-" * 120)
    query = 'Generate an index with label ESG from January 1st, 1984, to December 31st, 2021.'
    generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    index = generate.generate_emb()
    topic = Topic(index=index, score_col='score', article_col='body_txt', top_n=100, n_topic=20, output="esg_index")
    topic.get_topic()

    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Generate an index with label US Economic Policy Uncertainty from January 1st, 1984, to December 31st, 2021.'
    # generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    # index = generate.generate_emb()
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Generate an index with label US-China Trade War from January 1st, 1984, to December 31st, 2021.'
    # generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    # index = generate.generate_emb()
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Generate an index with label Artificial Intelligence from January 1st, 1984, to December 31st, 2021.'
    # generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    # index = generate.generate_emb()
    # index = generate.join_index(index=index, file_path='ai_google_trend.parquet.brotli')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Generate an index with label Blockchain from January 1st, 1984, to December 31st, 2021.'
    # generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    # index = generate.generate_emb()
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Generate an index with label COVID-19 from January 1st, 1984, to December 31st, 2021.'
    # generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    # index = generate.generate_emb()
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Generate an index with label Economic Recession from January 1st, 1984, to December 31st, 2021.'
    # generate = GenEmb(query=query, vector_data=wsj_openai, vector_column=vector_column, article_data=wsj_art, interval=interval, threshold=threshold)
    # index = generate.generate_emb()
