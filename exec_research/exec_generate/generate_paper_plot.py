import os
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

from utils.system import *
from class_data.data import Data
from class_generate.gen_emb import GenEmb

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------GET PLOTS---------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------PARAMS-----------------------------------------------------------------------
    # Params
    type = 'wsj'
    expand = True
    info = False
    vector_column = 'ada_embedding'
    interval = 'M'
    p_val = 0.01

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------WSJ-----------------------------------------------------------------------
    if type == 'wsj':
        # Load openai embeddings
        vector_data = Data(folder_path=get_format_data() / 'openai', file_pattern='wsj_emb_textemb3small_*')
        vector_data = vector_data.concat_files()

        # Load articles
        article_data = Data(folder_path=get_format_data() / 'token', file_pattern='wsj_tokens_*')
        article_data = article_data.concat_files()

        # Set limit for number of articles per date
        limit = 30
        count = vector_data.groupby(vector_data.index)[vector_data.columns[0]].count()
        valid_date = count >= limit
        vector_data = vector_data[vector_data.index.isin(count[valid_date].index)]
        count = article_data.groupby(article_data.index)[article_data.columns[0]].count()
        valid_date = count >= limit
        article_data = article_data[article_data.index.isin(count[valid_date].index)]

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------CC-----------------------------------------------------------------------
    elif type == 'cc':
        # Load CC openai embeddings
        vector_data = Data(folder_path=get_format_data() / 'openai', file_pattern='cc_emb_textemb3small_*')
        vector_data = vector_data.concat_files()

        # Load CC articles
        article_data = Data(folder_path=get_format_data() / 'token', file_pattern='cc_tokens_*')
        article_data = article_data.concat_files()

        # Daily Multiple CC Metadata
        mdata = Data(folder_path=get_data() / 'cc_multiple', file_pattern='*_mdata.pq')
        mdata = mdata.concat_files()

        # Create date index
        mdata['date'] = pd.to_datetime(mdata['startDate'], format='%d-%b-%y %I:%M%p %Z')
        mdata['date'] = mdata['date'].dt.date
        mdata['date'] = pd.to_datetime(mdata['date'])
        mdata = mdata.set_index('fid')

        # Set index for CC embeddings
        vector_data.index = article_data['fid']
        vector_data = vector_data.join(mdata)
        vector_data = vector_data.reset_index().set_index('date').sort_index()
        vector_data = vector_data[['ada_embedding']]
        vector_data = vector_data.loc[~vector_data.ada_embedding.isnull()]

        # Set index for CC articles
        article_data = article_data.set_index('fid')
        article_data = article_data.join(mdata)
        article_data = article_data.rename(columns={'Headline': 'headline'})
        article_data = article_data.reset_index().set_index('date').sort_index()
        article_data = article_data.loc[~((article_data.index == '2006-10-18') & (article_data.fid == '1391246') & (article_data.content_type == 'Presentation'))]
        article_data = article_data[['headline', 'body_txt']]

    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'ESG'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index = generate._join_index(index=index, interval=interval, file_path='esg_google_trend.parquet.brotli')
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['ESG', 'ESG (Google Trend)'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_esg_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'US Economic Policy Uncertainty'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index, pearson_corr = generate._compare_index(index=index, interval=interval, file_path='epu.parquet.brotli')
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=pearson_corr, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['US EPU', 'US EPU (Baker et al.)'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_usepu_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Inflation'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index, pearson_corr = generate._compare_index(index=index, interval=interval, file_path='ir.parquet.brotli')
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=pearson_corr, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Inflation', '5-Year Breakeven Inflation Rate (FRED)'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_inflation_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Economic Recession'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index, pearson_corr = generate._compare_index(index=index, interval=interval, file_path='recession.parquet.brotli')
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=pearson_corr, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Economic Recession', 'Economic Recession (Bybee et al.)'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_economicrecession_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Market Crash'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index = generate._standardize(index)
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Market Crash'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_marketcrash_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Stock Market Bubble'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index = generate._standardize(index)
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Stock Market Bubble'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_stockmarketbubble_index')

    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'US-China Trade War'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index = generate._standardize(index)
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['US-China Trade War'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_uschinatradewar_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Artificial Intelligence'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index = generate._standardize(index)
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Artificial Intelligence'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_ai_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Blockchain'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index = generate._standardize(index)
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Blockchain'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_blockchain_index')
    #
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'COVID-19'
    # generate = GenEmb(query=query, expand=False, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index = generate._standardize(index)
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['COVID-19'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_covid19_index')

    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Systemic Financial Stress'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index, pearson_corr = generate._compare_index(index=index, interval=interval, file_path='fsi.parquet.brotli')
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=pearson_corr, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Financial Stress', 'Financial Stress (Baker et al.)'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_fsi_index')

    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # # query = 'Volatility Index'
    # # query = 'Fear Index'
    # query = 'Option Volatility Index'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # index, pearson_corr = generate._compare_index(index=index, interval=interval, file_path='vix_option.parquet.brotli')
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=pearson_corr, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Option Volatility', 'CBOE Volatility (FRED)'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_vix_index')

    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # print("-" * 120)
    # query = 'Electric Vehicles'
    # generate = GenEmb(query=query, expand=expand, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    # index, expanded_query, threshold = generate.generate_emb()
    # # index, pearson_corr = generate._compare_index(index=index, interval=interval, file_path='tsla.parquet.brotli')
    # index = generate._standardize(index)
    # # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=pearson_corr, index_paper=index[['relu_score', 'official']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Electric Vehicles', 'TSLA Price (WRDS)'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_ev_index')
    # generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=pearson_corr, index_paper=index[['relu_score']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Electric Vehicles'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_ev_index')

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-"*120)
    query = 'Information Theory'
    generate = GenEmb(query=query, expand=False, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    index, expanded_query, threshold = generate.generate_emb()
    index = generate._standardize(index)
    generate.save(query=query, expanded_query=expanded_query, p_val=p_val, threshold=threshold, pearson=0, index_paper=index[['relu_score']], index_research=index[['relu_score', 'relu_norm_score', 'agg_norm_score', 'norm_score']], index_name_paper=['Information Theory'], index_name_research=['Transformed (Relu then Agg)', 'Transformed (Agg then Relu)', 'Non-Transformed (Agg)', 'Non-Transformed (No Agg)'], output=f'{type}_informationtheory_index')
    index[['relu_score']].to_csv(f'../../view_attention/{type}_informationtheory_index/monthly_index.csv')
    generate.daily_before_agg_index.to_csv(f'../../view_attention/{type}_informationtheory_index/daily_before_agg_index.csv')
    generate.daily_after_agg_index.to_csv(f'../../view_attention/{type}_informationtheory_index/daily_after_agg_index.csv')

    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # Get language data for research paper
    # def language_data(language, output_dir):
    #     os.makedirs(output_dir, exist_ok=True)
    #     label_collect = dict(English=1)
    #     time_series_collect = dict(English=1)
    #     for i, (lan, query) in enumerate(language):
    #         generate = GenEmb(query=query, expand=False, info=info, vector_data=vector_data, vector_column=vector_column, article_data=article_data, interval=interval, p_val=p_val)
    #         index, expanded_query, threshold = generate.generate_emb()
    #         if lan == 'English':
    #             eng_index = index
    #             eng_label = pd.Series(generate.label_emb.squeeze())
    #             continue
    #
    #         label_collect[f'{lan}'] = eng_label.corr(pd.Series(generate.label_emb.squeeze()))
    #         time_series_collect[f'{lan}'] = eng_index['relu_score'].corr(index['relu_score'])
    #
    #     df = pd.DataFrame({
    #         'Language': ['English'] + [lan for lan in label_collect if lan != 'English'],
    #         'Query': ['inflation'] + [f'{lan[1]}' for lan in language if lan[0] != 'English'],
    #         'Query Embedding Correlation': [1] + [label_collect[lan] for lan in label_collect if lan != 'English'],
    #         'Attention Index Correlation': [1] + [time_series_collect[lan] for lan in time_series_collect if lan != 'English']
    #     })
    #
    #     df.to_csv(output_dir + '/language.csv', index=False)
    #     print(df)
    #
    # print("-"*120)
    # language = [('English', 'Inflation'), ('Chinese', '通货膨胀'), ('Russian', 'инфляция'), ('Spanish', 'inflación'), ('French', "d'inflation"), ('Arabic', 'زِيادة في الأَسْعار')]
    # output_dir = '../../view_attention/language_inflation'
    # language_data(language=language, output_dir=output_dir)
    #
    # print("-"*120)
    # language = [('English', 'Economic Recession'), ('Chinese', '经济衰退'), ('Russian', 'экономическая рецессия'), ('Spanish', 'recesión económica'), ('French', 'récession économique'), ('Arabic', 'الركود الاقتصادي')]
    # output_dir = '../../view_attention/language_economicrecession'
    # language_data(language=language, output_dir=output_dir)
    #
    # print("-" * 120)
    # language = [('English', 'Artificial Intelligence'), ('Chinese', '人工智能'), ('Russian', 'искусственный интеллект'), ('Spanish', 'inteligencia artificial'), ('French', 'intelligence artificielle'), ('Arabic', 'الذكاء الاصطناعي')]
    # output_dir = '../../view_attention/language_ai'
    # language_data(language=language, output_dir=output_dir)