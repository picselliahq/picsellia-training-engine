from flask import Flask
from flask import request
import json
import requests
from flask import Response
from flask import jsonify
# import cv2
import time
from flask_cors import CORS
import os
import subprocess
import shlex


### TU METS TON ENDPOINT DANS LAUNCH TRAINING ET ON EST BON
def create_app():
  app = Flask(__name__)
  CORS(app)

  @app.route('/launch_training', methods=['POST'])
  def launch_training():
    api_token = request.json["api_token"]
    experiment_id = request.json["experiment_id"]
    command = "python3 launch_sub.py {} {}".format(api_token, experiment_id)
    subprocess.Popen(shlex.split(command))
    return "hello world"
    
  @app.route("/")
  def hello():

    subprocess.Popen(["python3","ping.py"])
    return "Hello World!"

  return app


if __name__ == "__main__":
  app = create_app()
  app.run(host='0.0.0.0', port='5000')
