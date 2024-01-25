import json

from openai import OpenAI

from utils.system import get_config

class GPTExtract:
    def __init__(self,
                 input=None):

        '''
        input (str): Input text to extract information from
        '''

        self.input = input
        self.api_key = json.load(open(get_config() / 'api.json'))['openai_api_key']


    def gpt_extract(self):
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"\n\n Here is the text: {input}" +
                                            "Can you extract the label, start date, end date, and transform from this piece of text. " +
                                            "The transform should exist in this list: ['relu', 'squared relu', 'arcsin', 'sigmoid']. " +
                                            "You must extract any variation of these four transformations. " +
                                            "For example, if the text is 'Rectified Linear Unit transformation', then store it as 'relu'. " +
                                            "If a transformation is not in the list when abbreviated or cut down, then store the transform as transform as 'none'. " +
                                            "Likewise, if you cannot extract a label, start date, or end date, then store the them as 'none'. " +
                                            "Your output should not contain any additional comments or add-ons. YOU MUST ONLY OUTPUT THIS: " +
                                            "{\"label\": \"(label)\", " +
                                            "\"start_date\": \"(YYYY-MM-DD)\", " +
                                            "\"end_date\": \"(YYYY-MM-DD)\", " +
                                            "\"transform\": \"(transform)\"}"
                 }
            ]
        )
        summary = response.choices[0].message.content.strip()
        data = json.loads(summary)
        return data