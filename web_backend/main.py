import re
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
from class_parser.gpt_extract import GPTExtract

app = Flask(__name__)
CORS(app, resources={r"/generate_plot": {"origins": "http://localhost:3000"}})

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
        1. Start the flask server by cding to the directory that this file is located in and running this in powershell: python app.py
        2. To test the server: Invoke-WebRequest -Uri http://localhost:5000/generate_plot -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"input_str": "Label: artificial intelligence, Start Date: 2010-01-01, End Date: 2010-01-01, Transform: relu"}'
    '''

     # Extract and parse input string
    data = request.json
    input_str = data.get('input_str', '')

    # Use GPTExtract to retrieve the required information
    gpt_extract = GPTExtract(input=input_str)

    try:
        extracted_info = gpt_extract.gpt_extract()
    except:
        return jsonify({'error': 'Invalid input format, please follow the desired input message format.'}), 400


    label = extracted_info['label']
    start_date_str = extracted_info['start_date']
    end_date_str = extracted_info['end_date']
    transform = extracted_info['transform']

    if label == 'None' and start_date_str == 'None' and end_date_str == 'None' and transform == 'None':
        return jsonify({'error': 'Invalid input format, please clearly specify all required specifications and try again.'}), 400
    elif label == 'None':
        return jsonify({'error': 'A label was not specified or not correctly processed, please clearly specify a label and try again.'}), 400
    elif start_date_str == 'None':
        return jsonify({'error': 'A start date was not specified or not correctly processed, please clearly specify a start date and try again.'}), 400
    elif end_date_str == 'None':
        return jsonify({'error': 'An end date was not specified or not correctly processed, please clearly specify an end date and try again.'}), 400
    elif transform == 'None':
        return jsonify({'error': 'A transformation was not specified or not correctly processed, please clearly specify a transformation and try again.'}), 400

    # Check the format of start_date and end_date
    date_format = r"\d{4}-\d{2}-\d{2}"
    if not (re.match(date_format, start_date_str) and re.match(date_format, end_date_str)):
        return jsonify({'error': 'Invalid date format, please use YYYY-MM-DD for start_date and end_date.'}), 400

    # Convert start_date and end_date to date objects
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format, please use YYYY-MM-DD for start_date and end_date.'}), 400

    # Check if transform is one of the specified values
    valid_transforms = ['relu', 'square relu', 'arcsin', 'sigmoid']
    if transform not in valid_transforms:
        return jsonify({'error': 'Invalid transform value, valid options are relu, square relu, arcsin, sigmoid.'}), 400

    # Check if the date is within the specified range
    min_date = datetime(1980, 1, 1).date()
    max_date = datetime(2022, 12, 31).date()
    if not (min_date <= start_date <= max_date) or not (min_date <= end_date <= max_date):
        return jsonify({'error': 'Invalid date value, date should be between 1980 and 2022.'}), 400

    # Check if start_date is before end_date
    if start_date >= end_date:
        return jsonify({'error': 'Invalid date value, start_date should be before end_date.'}), 400

    # Fetch Data from Milvus Database
    query_fetch = QueryFetch(label=label, start_date=start_date_str, end_date=end_date_str)
    query = query_fetch.query_fetch()

    # Generate Index and Article index
    generate_index = GenerateIndex(query=query, transform=transform)
    gen_index, gen_combine = generate_index.generate_index()

    # Plot Index
    plot_plotly = PlotPlotly(data=gen_combine)
    plot_fig = plot_plotly.get_plot()

    return jsonify({
        'gen_plot': plot_fig,
        'gen_index': gen_index.to_csv(),
        'gen_combine': gen_combine.to_csv()
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
    # # Local
    # app.run(debug=True)

    # Prod (GCP)
    app.run(host="0.0.0.0", port=5000)
