import subprocess
import shlex
from picsellia.client import Client
import os
import re 
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
from picsellia.pxl_exceptions import AuthenticationError

command = "python3 picsellia/docker_run_training_tf2.py"
host = 'https://beta.picsellia.com/sdk/v2/'
if 'api_token' not in os.environ:
    raise AuthenticationError("You must set an api_token to run this image")

api_token = os.environ['api_token']

if "experiment_id" in os.environ:
    experiment_id = os.environ['experiment_id']

    experiment = Client.Experiment(api_token=api_token)
    exp = experiment.checkout(experiment_id)
else:
    if "experiment_name" in os.environ and "project_token" in os.environ:
        project_token = os.environ['project_token']
        experiment_name = os.environ['experiment_name']
        experiment = Client.Experiment(api_token=api_token, project_token=project_token)
        exp = experiment.checkout(experiment_name)
    else:
        raise AuthenticationError("You must either set the experiment id or the project token + experiment_name")


process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
part = "--#--Set up training"
replace_log = False
buffer = []
start_buffer = False
buffer_length = 0
while True:
    output = process.stdout.readline()
    if output.decode("utf-8")  == '' and process.poll() is not None:
        break
    print(output.decode("utf-8")) 
    if output:
        if output.decode("utf-8").startswith('--#--'):
            part = output.decode("utf-8")

        if output.decode("utf-8").startswith('-----'):
            progress_line_nb = exp.line_nb
            replace_log = True

        if output.decode("utf-8").startswith('--*--'):
            replace_log = False

        if re.match("--[0-9]--", output.decode("utf-8")[:6]):
            start_buffer = True
            buffer_length = int(output.decode("utf-8")[2])

        if re.match("---[0-9]---", output.decode("utf-8")[:8]):
            start_buffer = False
            exp.send_experiment_logging(buffer, part, special='buffer')
            exp.line_nb += (len(buffer)-1)
            buffer = []

        if start_buffer:
            buffer.append(output.decode("utf-8"))
            if len(buffer)==buffer_length:
                exp.send_experiment_logging(buffer, part, special='buffer')
                exp.line_nb += (buffer_length-1)
                buffer = []
        else:
            if not replace_log:
                exp.send_experiment_logging(output.decode("utf-8"), part)
            else:
                exp.line_nb = progress_line_nb
                exp.send_experiment_logging(output.decode("utf-8"), part)
        
if buffer != []:
    exp.send_experiment_logging(buffer, part, special='buffer')
exp.send_experiment_logging(str(process.returncode), part, special='exit_code')
# if process.returncode == 0 or process.returncode == "0":
#     exp.update_experiment_status('succeeded')
# else:
#     clt.update_experiment_status('failed')
rc = process.poll()
