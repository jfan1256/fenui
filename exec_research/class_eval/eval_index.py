import json
import time
import numpy as np
import seaborn as sns

from concurrent.futures import ThreadPoolExecutor, as_completed

from matplotlib import pyplot as plt
from openai import OpenAI, RateLimitError
from tqdm import tqdm

from class_data.data import Data
from class_generate.gen_index import GenIndex
from utils.system import get_config, get_format_data


class EvalIndex:
    def __init__(self,
                 index=None,
                 label=None,
                 art_col=None,
                 eval_col=None,
                 batch_size=None):
        
        '''
        index (pd.DataFrame): Pandas dataframe containing the score, headline, and article
        label (str): Label that the index was generated based off of
        art_col (str): Column name of the article column
        eval_col (str): Column name of the generated eval column
        batch_size (int): Batch size for parallelization
        '''
        
        self.index = index
        self.label = label
        self.art_col = art_col
        self.eval_col = eval_col
        self.batch_size = batch_size

    def eval(self, article_text):
        api_key = json.load(open(get_config() / 'api.json'))['openai_api_key']
        client = OpenAI(api_key=api_key)
        retry_attempts = 5
        retry_wait = 0.5

        # Get first 200 words
        words = article_text.split()
        article_text = words[:200]
        article_text = ' '.join(article_text)

        for attempt in range(retry_attempts):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": f"Only output a python str of 1, 2, 3, 4, or 5 where larger number means more relevant for this question\n\n: Is this piece of text related to {self.label}: \n\n{article_text}"}
                    ],
                    temperature=1,
                    max_tokens=150,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].message.content.strip()
            except RateLimitError as e:
                print(f"Rate limit exceeded, retrying in {retry_wait} seconds...")
                time.sleep(retry_wait)
                retry_wait *= 2
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        raise Exception("Failed to process the article after several retries.")

    def eval_articles(self):
        print("-" * 60)
        num_batches = np.ceil(len(self.index) / self.batch_size).astype(int)
        all_summary = [None] * len(self.index)
        print(f"Number of batches: {num_batches}")

        for batch_num in tqdm(range(num_batches), desc="Processing batch"):
            start_index = batch_num * self.batch_size
            end_index = min((batch_num + 1) * self.batch_size, len(self.index))
            batch = self.index.iloc[start_index:end_index]

            # Using ThreadPoolExecutor to process the current batch in parallel
            with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                future_to_index = {executor.submit(self.eval, article_text): i for i, article_text in enumerate(batch[self.art_col].tolist(), start=start_index)}

                # Ensure results are stored in the original order within the batch
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    all_summary[index] = future.result()

        # Assigning the computed summaries back to the DataFrame
        self.index[self.eval_col] = all_summary
        return self.index

    def count(self, eval):
        # Print the counts for each rank
        print("-" * 60)
        rank = ['1', '2', '3', '4', '5']
        for r in rank:
            col = eval.loc[eval[self.eval_col] == r]
            print(f"Number of {r}: {len(col)}/{len(eval)}")

        score_column = 'score'

        # Plot setup
        plt.figure(figsize=(10, 4))
        sns.set(style="whitegrid")

        # Plotting the distribution of scores for each rank
        for r in rank:
            subset = eval[eval[self.eval_col] == r][score_column]
            if subset.var() == 0:
                print(f"Skipping Rank {r} due to 0 variance")
                continue
            sns.kdeplot(subset, fill=True, label=f'Rank {r}', common_norm=False)

        # Finalizing the plot
        plt.title('Distribution of Scores by Rank')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend(title='Rank')
        plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------TEST--------------------------------------------------------------------------
if __name__ == "__main__":
    # Load openai embeddings
    wsj_openai = Data(folder_path=get_format_data() / 'openai', file_pattern='wsj_emb_openai_*')
    wsj_openai = wsj_openai.concat_files(10)

    # Load articles
    wsj_art = Data(folder_path=get_format_data() / 'token', file_pattern='wsj_tokens_*')
    wsj_art = wsj_art.concat_files(1)

    # Equal data
    wsj_openai = wsj_openai.head(10000)
    wsj_art = wsj_art.head(10000)

    # Params
    type = 'embedding'
    vector_column = 'ada_embedding'
    interval = 'M'

    # Generate
    query = 'Generate an index with label ESG from 1984-01-02 to 2021-12-31'
    generate = GenIndex(query=query,
                        type=type,
                        vector_data=wsj_openai,
                        vector_column=vector_column,
                        article_data=wsj_art,
                        interval=interval,
                        threshold=0.77,
                        alpha=0.01)
    esg = generate.generate_emb()

    # Evaluate
    eval_index = EvalIndex(index=esg, label=generate.query['label'], art_col='body_txt', eval_col='eval', batch_size=1)
    eval_esg = eval_index.eval_articles()
    eval_index.count(eval_esg)


    
    