import os
import json
import time
import uuid
import config
import traceback
import pandas as pd

from flask_cors import CORS
from datetime import datetime
from flask import Flask, request, jsonify, g

from class_data.data import Data
from utils.system import get_format_data
from class_plot.plot_plotly import PlotPlotly
from class_generate.generate_index import GenerateIndex
from class_expand.expand_query import ExpandQuery

app = Flask(__name__)
prod = False

if prod == True:
    # Prod CORS (This is connecting to the frontend url)
    CORS(app, resources={r"/generate_plot": {"origins": "https://fenui.vercel.app"}})
else:
    # Local CORS
    CORS(app, resources={r"/generate_plot": {"origins": "http://localhost:3000"}})

# Global data storage
print("-" * 60 + "\nLoading data into memory")
# Multiple Articles per Day Open AI Embeddings
data = Data(folder_path=get_format_data() / 'web', file_pattern='wsj_all_*')
global_data = data.concat_files()
global_data = global_data.reset_index().set_index('date')
global_data = global_data[['ada_embedding', 'headline', 'body_txt']]
print("-" * 60 + "\nData loaded successfully")

# Log Version
@app.route("/version", methods=["GET"], strict_slashes=False)
def version():
    response_body = {
        "success": 1,
    }
    return jsonify(response_body)

# Generate Plot
@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    '''
    To test local flask server run this:
        1. Start the flask server by cding to the directory that this file is located in and running this in powershell: python flask_generate_pq.py
        2. To test the server: Invoke-WebRequest -Uri http://localhost:5000/generate_plot -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"input_str": "Label: artificial intelligence, Start Date: 2010-01-01, End Date: 2010-01-01, P-val: 0.01"}'
    '''

     # Extract and parse input string
    data = request.json
    input_str = data.get('input_str', '')

    # Use GPTExtract to retrieve the required information
    expand_query = ExpandQuery(query=input_str)

    try:
        extracted_info = expand_query.execute()
    except Exception as e:
        print("-" * 60 + f"\n{traceback.format_exc()}")
        return jsonify({'error': 'Invalid input format, please follow the desired input message format.'}), 400

    # Store extracted info
    query = extracted_info['expanded_query']
    start_date_str = extracted_info['start_date']
    end_date_str = extracted_info['end_date']
    p_val = extracted_info['p_val']

    # Check query
    if query in [None, 'None', 'null'] :
        return jsonify({'error': 'A query was not specified or not correctly processed, please clearly specify a query and try again.'}), 400

    # Convert start_date and end_date to date objects
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except Exception as e:
        print("-" * 60 + f"\n{traceback.format_exc()}")
        return jsonify({'error': 'Invalid date format, please use YYYY-MM-DD for start_date and end_date.'}), 400

    # Check if the date is within the specified range
    min_date = datetime(1980, 1, 1).date()
    max_date = datetime(2022, 12, 31).date()
    if not (min_date <= start_date <= max_date) or not (min_date <= end_date <= max_date):
        return jsonify({'error': 'Invalid date value, date should be between 1980 and 2022.'}), 400

    # Check if start_date is before end_date
    if start_date >= end_date:
        return jsonify({'error': 'Invalid date value, start_date should be before end_date.'}), 400

    # Get Data and calculate score
    try:
        data = global_data.loc[(global_data.index >= start_date_str) & (global_data.index <= end_date_str)]
    except Exception as e:
        print("-" * 60 + f"\n{traceback.format_exc()}")
        return jsonify({'error': 'Unable to retrieve data, please try again soon or later.'}), 400

    # Generate Index and Article index
    print("-" * 60 + f"\nGenerate Index")
    generate_index = GenerateIndex(data=data, query=query, p_val=p_val)
    gen_index, gen_combine = generate_index.generate_index_pq()
    print("-" * 60 + f"\nGenerated Dataframe: \n\n\n{gen_combine}")

    # Plot Index
    print("-" * 60 + f"\nPlot Index")
    plot_plotly = PlotPlotly(data=gen_index)
    plot_fig = plot_plotly.get_plot()

    # Save expanded query into pandas dataframe
    expand_query = pd.DataFrame([extracted_info])

    return jsonify({
        'gen_plot': plot_fig,
        'gen_index': gen_index.to_csv(),
        'gen_combine': gen_combine.to_csv(),
        'expand_query': expand_query.to_csv(index=False)
    })

# Log Version
@app.after_request
def after_request(response):
    if response and response.get_json():
        data = response.get_json()
        data["time_request"] = int(time.time())
        data["version"] = config.VERSION
        response.set_data(json.dumps(data))
    return response

# Execution ID
@app.before_request
def before_request_func():
    execution_id = uuid.uuid4()
    g.start_time = time.time()
    g.execution_id = execution_id
    print(g.execution_id, "ROUTE CALLED ", request.url)

if __name__ == "__main__":
    if prod == True:
        # Prod (GCP or Ngrok)
        app.run(host="0.0.0.0", port=5000)
    else:
        # Local
        app.run(debug=True)


