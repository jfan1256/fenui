This folder contains all the data used in this research paper. 

Here is the folder structure as well as its information:
├───cc (contains conference call embeddings)
├───cc_multiple (contains raw conference call data (i.e., metatdata, text, etc.))
├───format (contains all the formatted and preprocessed data via the Data class found under /exec_research/class_data/data.py)
│   ├───art (contains compressed raw WSJ article files)
│   ├───cosine_sim (contains test cosine similarity indexes)
│   ├───eval (contains test evaluation indexes)
│   ├───mxbai (contains embeddings of WSJ generated via Mxbai LLM)
│   ├───openai (contains embeddings of WSJ generated via OpenAI LLM)
│   ├───preprocess (contains preprocessed WSJ articles)
│   ├───tfidf (contains tfidf 
│   ├───token (contains token count WSJ article files)
│   └───web (contains WSJ article files and embeddings used for website)
├───nyt (contains NYT embeddings)
├───wsj_multiple (contains raw WSJ articles files)
└───wsj_single (contains single daily WSJ articles files)


The other files in this folder (either .csv or .xlsx) are the raw indexes downloaded from various sources online (i.e., FRED, OFR, etc.). These are all processed, formatted, and saved in the "format" directory. 
