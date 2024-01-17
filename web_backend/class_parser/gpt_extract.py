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
                {"role": "user", "content": "Can you extract the label, start date, end date, and transform from this piece of text."
                                            "The transform should exist in this list: ['relu', 'squared relu', 'arcsin', 'sigmoid']. I would like you to be able to extract"
                                            "any name variation of these transformations. So if for example "
                                            "the user specifies a Rectified Linear Unit transformation, log it as 'relu'. If a transformation is not in the list when"
                                            "abbreviated or cut down, then input transform as none. "
                                            "In addition, please format the data like a json all lowercase string file like this:"
                                            "\n\n label: (label), start date: (YYYY-MM-DD), end date: (YYYY-MM-DD), transform: (transform) "
                                            "\n\n if you cannot find a label, start date, end date, or a transform, set the value to be 'None' within the json output"
                                            f"\n\n Here is the text: {self.input}"
                 }
            ]
        )
        summary = response.choices[0].message.content.strip()
        data = json.loads(summary)
        extracted_info = {
            'label': data.get('label'),
            'start_date': data.get('start date'),
            'end_date': data.get('end date'),
            'transform': data.get('transform')
        }

        return extracted_info