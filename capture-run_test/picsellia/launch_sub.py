import subprocess
import shlex
from picsellia.client import Client
import os
import re 
os.environ["PYTHONUNBUFFERED"] = "1"
os.chdir('picsellia')
import sys
from picsellia.pxl_exceptions import AuthenticationError
from datetime import date, datetime
import json

host = 'http://127.0.0.1:8000/sdk/v2/'

api_token = 'ac7a44b7be181774bd088c0099afd449b26bbeb7'
project_token = '3d039f84-7224-4442-b499-652f2f77f1a1'
experiment_name = "test2"
experiment = Client.Experiment(api_token=api_token, project_token=project_token, host=host)

exp = experiment.checkout(name=experiment_name)
os.environ["experiment_id"] = exp.id
os.environ["api_token"] = api_token
os.environ["project_token"] = project_token


command = "python3 {}/{}".format(os.getcwd(), 'test_file.py')
process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
part = "--#--Start Run"
replace_log = False
buffer = []
start_buffer = False
buffer_length = 0
exp.send_experiment_logging(part, part)

last_line = ""
logs = {}
logs[part] = {
    'datetime': str(datetime.now().isoformat()),
    'logs': {}
}
while True:
    output = process.stdout.readline()
    if output.decode("utf-8")  == '' and process.poll() is not None:
        break
    print(output.decode("utf-8")) 
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
