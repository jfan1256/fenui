import re
import traceback

import pandas as pd

import config
import json
import time
import uuid

from flask import Flask, request, jsonify, g
from datetime import datetime
from flask_cors import CORS

from class_fetch.query_fetch import QueryFetch
from class_generate.generate_index import GenerateIndex
from class_plot.plot_plotly import PlotPlotly
from class_expand.expand_query import ExpandQuery

app = Flask(__name__)

prod = False

if prod == True:
    # Prod CORS (This is connecting to the frontend url)
    CORS(app, resources={r"/generate_plot": {"origins": "https://fenui.vercel.app"}})

    # Ngrok Tunnel (This is a secure tunnel to broadcast localhost to the internet - for milvus that would be broadcasting localhost:19530)
    # Must update this everytime by running ngrok tcp 19530 in ngrok command prompt
    # Must ensure that milvus docker-compose is running on-prem
    ngrok_config = json.load(open('../ngrok/ngrok.json'))
    ngrok_host = ngrok_config['tcp_host']
    ngrok_port = ngrok_config['tcp_port']

else:
    # Local CORS
    CORS(app, resources={r"/generate_plot": {"origins": "http://localhost:3000"}})
    ngrok_host = None
    ngrok_port = None

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
        1. Start the flask server by cding to the directory that this file is located in and running this in powershell: python main.py
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


    query = extracted_info['expanded_query']
    start_date_str = extracted_info['start_date']
    end_date_str = extracted_info['end_date']
    p_val = extracted_info['p_val']

    # Check query
    if query == 'None':
        return jsonify({'error': 'A query was not specified or not correctly processed, please clearly specify a label and try again.'}), 400

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

    # Fetch Data from Milvus Database
    try:
        query_fetch = QueryFetch(label=query, start_date=start_date_str, end_date=end_date_str, prod=prod, ngrok_host=ngrok_host, ngrok_port=ngrok_port)
        query = query_fetch.query_fetch()
    except Exception as e:
        print("-" * 60 + f"\n{traceback.format_exc()}")
        return jsonify({'error': 'Unable to connect to database, please try again soon or later.'}), 400
    print("-" * 60 + f"\nmilvus data: {query}")

    # Generate Index and Article index
    print("-" * 60 + f"\nGenerate Index")
    generate_index = GenerateIndex(query=query, p_vaL=p_val)
    gen_index, gen_combine = generate_index.generate_index()
    print("-" * 60 + f"\ngen_index: {gen_index}")
    print("-" * 60 + f"\ngen_combine: {gen_combine}")

    # Plot Index
    print("-" * 60 + f"\nPlot Index")
    plot_plotly = PlotPlotly(data=gen_index)
    plot_fig = plot_plotly.get_plot()

    # Save expanded query into pandas dataframe
    expand_query = pd.DataFrame([extracted_info])
    print("-" * 60 + f"\nexpand_query: {expand_query}")

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


