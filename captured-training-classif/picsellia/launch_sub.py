import subprocess
import shlex
from picsellia.client import Client
import os
import re 
import json
os.environ["PYTHONUNBUFFERED"] = "1"
os.chdir('picsellia')
from datetime import datetime
from picsellia.pxl_exceptions import AuthenticationError

command = "python3 docker_run_training_classif.py"
if "host" not in os.environ:
    host = "https://app.picsellia.com/sdk/v2/"
else:
    host = os.environ["host"]
if 'api_token' not in os.environ:
    raise AuthenticationError("You must set an api_token to run this image")
api_token = os.environ["api_token"]

if "experiment_id" in os.environ:
    experiment_id = os.environ['experiment_id']

    experiment = Client.Experiment(api_token=api_token, host=host)
    exp = experiment.checkout(experiment_id)
else:
    if "experiment_name" in os.environ and "project_token" in os.environ:
        project_token = os.environ['project_token']
        experiment_name = os.environ['experiment_name']
        experiment = Client.Experiment(api_token=api_token, project_token=project_token, host=host)
        exp = experiment.checkout(experiment_name)
    else:
        raise AuthenticationError("You must either set the experiment id or the project token + experiment_name")


process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
part = "--#--Set up training"
replace_log = False
buffer = []
start_buffer = False
buffer_length = 0
exp.send_experiment_logging(part, part)
logs = {}
logs[part] = {
    'datetime': str(datetime.now().isoformat()),
    'logs': {}
}
last_line = ""
while True:
    output = process.stdout.readline()
    if output.decode("utf-8")  == '' and process.poll() is not None:
        break
    text = output.decode("utf-8")
    if output:
        if text.startswith('--#--'):
            part = output.decode("utf-8")
            logs[part] = {
                'datetime': str(datetime.now().isoformat()),
                'logs': {}
            }
        if text.startswith('-----'):
            progress_line_nb = exp.line_nb
            replace_log = True

        if text.startswith('--*--'):
            replace_log = False

        if re.match("--[0-9]--", text[:6]):
            start_buffer = True
            buffer_length = int(text[2])

        if re.match("---[0-9]---", text[:8]):
            start_buffer = False
            exp.send_experiment_logging(buffer, part, special='buffer')
            exp.line_nb += (len(buffer)-1)
            buffer = []

        if start_buffer:
            buffer.append(text)
            logs[part]['logs'][str(exp.line_nb+len(buffer))] = text
            if len(buffer)==buffer_length:
                exp.send_experiment_logging(buffer, part, special='buffer')
                exp.line_nb += (buffer_length-1)
                buffer = []
        else:
            if not replace_log:
                exp.send_experiment_logging(text, part)
                logs[part]['logs'][str(exp.line_nb)] = text
            else:
                exp.line_nb = progress_line_nb
                exp.send_experiment_logging(text, part)
        
        last_line = text

        
with open('{}-logs.json'.format(exp.id), 'w') as f:
    if buffer != []:
        for i, line in enumerate(buffer):
            logs[part]['logs'][str(exp.line_nb+i)] = line
        exp.send_experiment_logging(buffer, part, special='buffer')
    logs["exit_code"] = {
        'exit_code': str(process.returncode),
        'datetime': str(datetime.now().isoformat())
    }
    json.dump(logs, f) 
exp.send_experiment_logging(str(process.returncode), part, special='exit_code')
exp.store('logs','{}-logs.json'.format(exp.id))

if process.returncode == 0 or process.returncode == "0":
    exp.update(status='success')
else:
    exp.update(status='failed')
rc = process.poll()
