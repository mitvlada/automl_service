from flask import Flask
import requests
import traceback
import logging
from datetime import datetime

from autoML_service.autoML.session import AutoMLSession

app = Flask(__name__)

error_message = "ERROR!<br><br> Runtime error occured. Check log from Docker container. Please refresh page manualy.<br><br>"
no_response_message =  """<meta http-equiv="refresh" content="5"/>NO MESSAGE!<br><br> The task is not configured from user side. The page will refresh automatically in 5 seconds."""
model_trained_message = "SUCCESS!<br><br> The model is trained. Please refresh page manualy."


@app.route("/")
def home():

    response_json = {}
    url = "http://automl-service-user-side-container:5000/message"

    try:
        res = requests.get(url)
        response_json = res.json()
    except Exception as e:
        logging.error(traceback.format_exc())
        return no_response_message

    if not any(response_json.values()): 
        return no_response_message
    
    try:
        session = AutoMLSession(response_json)
        session.run_session()
        return model_trained_message
    except Exception as e:
        print("ERROR!")
        logging.error(traceback.format_exc())        
        return error_message


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
