import subprocess
import shlex
from picsellia import Client
import os
import re 
import json
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ['PICSELLIA_SDK_CUSTOM_LOGGING'] = "True" 
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
os.environ["PICSELLIA_SDK_SECTION_HANDLER"] = "1"

os.chdir('picsellia')
from datetime import datetime
from picsellia.types.enums import ExperimentStatus, JobStatus
import logging

logging.getLogger('picsellia').setLevel(logging.INFO)

command = "python3.10 docker_run_training_yolov8_classification.py"

if "host" not in os.environ:
    host = "https://app.picsellia.com"
else:
    host = os.environ["host"]
if 'api_token' not in os.environ:
    raise RuntimeError("You must set an api_token to run this image")
api_token = os.environ["api_token"]

client = Client(
    api_token=api_token,
    host=host
)

if "experiment_name" in os.environ:
    experiment_name = os.environ["experiment_name"]
    if "project_token" in os.environ:
        project_token = os.environ["project_token"]
        project = client.get_project_by_id(project_token)
    elif "project_name" in os.environ:
        project_name = os.environ["project_name"]
        project = client.get_project(project_name)
    experiment = project.get_experiment(experiment_name)
else:
    raise Exception("You must set the project_token or project_name and experiment_name")

process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
part = "--#--Set up training"
replace_log = False
buffer = []
start_buffer = False
buffer_length = 0
try:
    experiment.send_logging(part, part)
except Exception:
    pass
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
            progress_line_nb = experiment.line_nb
            replace_log = True

        if text.startswith('--*--'):
            replace_log = False

        if re.match("--[0-9]--", text[:6]):
            start_buffer = True
            buffer_length = int(text[2])

        if re.match("---[0-9]---", text[:8]):
            start_buffer = False
            try:
                experiment.send_logging(buffer, part, special='buffer')
                experiment.line_nb += (len(buffer)-1)
            except Exception:
                pass
            buffer = []

        if start_buffer:
            buffer.append(text)
            logs[part]['logs'][str(experiment.line_nb+len(buffer))] = text
            if len(buffer)==buffer_length:
                try:
                    experiment.send_logging(buffer, part, special='buffer')
                    experiment.line_nb += (buffer_length-1)
                except Exception:
                    pass
                buffer = []
        else:
            if not replace_log:
                try:
                    experiment.send_logging(text, part)
                    logs[part]['logs'][str(experiment.line_nb)] = text
                except Exception:
                    pass
            else:
                try:
                    experiment.line_nb = progress_line_nb
                    experiment.send_logging(text, part)
                except Exception:
                    pass
        
        last_line = text

        
with open('{}-logs.json'.format(experiment.id), 'w') as f:
    if buffer != []:
        for i, line in enumerate(buffer):
            logs[part]['logs'][str(experiment.line_nb+i)] = line
        experiment.send_logging(buffer, part, special='buffer')
    logs["exit_code"] = {
        'exit_code': str(process.returncode),
        'datetime': str(datetime.now().isoformat())
    }
    json.dump(logs, f) 
experiment.send_logging(str(process.returncode), part, special='exit_code')
experiment.store_logging_file('{}-logs.json'.format(experiment.id))

if process.returncode == 0 or process.returncode == "0":
    experiment.update(status=ExperimentStatus.SUCCESS)
    experiment.update_job_status(status=JobStatus.SUCCESS)
else:
    experiment.update(status=ExperimentStatus.FAILED)
    experiment.update_job_status(status=JobStatus.FAILED)
rc = process.poll()