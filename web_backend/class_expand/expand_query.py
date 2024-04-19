import json

from openai import OpenAI

from utils.system import get_config

class ExpandQuery:
    def __init__(self,
                 query=None):

        '''
        input (str): Input text to extract information from
        '''

        self.query = query
        self.api_key = json.load(open(get_config() / 'api.json'))['openai_api_key']
        self.client = OpenAI(api_key=self.api_key)

    # Extract info
    def _extract_info(self):
        # Embedding Extraction
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": "Given a user's textual request about tracking the attention index of a specific topic or narrative over time, extract the relevant fields so that they can be supplied to function in a structured format."
                                            f"Here is the user’s request input: {self.query}"
                                            "The fields to extract are:"
                                            "query: The specific topic or narrative the user wants to track. If not specified, return null."
                                            "start_date: The start date for tracking, formatted as YYYY-MM-DD. If not specified, return null."
                                            "end_date: The end date for tracking, formatted as YYYY-MM-DD. If not specified, return null."
                                            "p_val: The p-value specified by the user. If not specified, return null."
                                            "expand: The boolean of whether to expand or not, formatted as true or false. If not specified, return null."
                                            "Return a JSON object with the keys query, start_date, end_date, p_val, and expand. If a field is missing, set the corresponding value to null. This will indicate that the field was not specified by the user and can be filled with default values later."
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
        print("-"*60 + f"\nExtract Info Query: {self.query}")

    # Expand Query
    def _expand_query(self):
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": "Your job is to output expanded queries to represent the concept in the input query:"
                                            f"'{self.query['query']}'"
                                            "In detail, I have over 800000 news articles."
                                            "I want to track how much each article pertains to the a topic or a narrative contained in the input query:"
                                            f"'{self.query['query']}'"
                                            "I am going to use openai's embedding model to compare the cosine similarity between each article and the list of expanded queries that encapsulate the concept in the input query."
                                            "Please transform the input query into an extensive set of queries that accurately, thoroughly, and vividly encompass all interpretations, perspectives, and facets of the input query."
                                            "Please output in JSON format as {expanded_queries: [expanded_query_1, expanded_query_2, expanded_query_3, …]}."
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
        summary = json.loads(summary)
        paragraph = '. '.join(summary['expanded_queries']) + '.'
        self.query['expanded_query'] = paragraph
        print("-"*60 + f"\nExpanded Query: {self.query}")

    def execute(self):
        # Extract info (i.e., start_date, end_date, etc.) or set default values
        self._extract_info()

        # Return if query is none
        if self.query['query'] in [None, 'null', 'None']:
            return self.query

        # Set default values
        if self.query['start_date'] in [None, 'null', 'None']:
            self.query['start_date'] = '1984-01-01'
        if self.query['end_date'] in [None, 'null', 'None']:
            self.query['end_date'] = '2021-12-31'
        if self.query['p_val'] in [None, 'null', 'None']:
            self.query['p_val'] = 0.01
        if self.query['expand'] in [None, 'null', 'None']:
            self.query['expand'] = True

        # Expand query
        if self.query['expand']:
            self._expand_query()
        else:
            self.query['expanded_query'] = self.query['query']
        return self.query