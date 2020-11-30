import subprocess
import shlex
from picsellia_training.client import Client
import os
import re 
os.environ["PYTHONUNBUFFERED"] = "1"
import sys

### Docker
# api_token = os.environ['api_token']
# experiment_id = os.environ['experiment_id']
# command = "python3 tf_training_od.py"
# host = 'https://demo.picsellia.com/sdk/'

### Local tf1
# api_token = "4d388e237d10b8a19a93517ffbe7ea32ee7f4787"
# experiment_id = '221586b8-4e04-47f7-bded-1cb502d31c01'
# command = "python tf_training_od.py {} {}".format(api_token, experiment_id)
# host = 'http://127.0.0.1:8000/sdk/'

### Local tf2
# api_token = "4d388e237d10b8a19a93517ffbe7ea32ee7f4787"
# experiment_id = '9bfbfa3c-eeaf-4268-ace5-360aabd861e6'
# command = "python tf_training_od.py {} {}".format(api_token, experiment_id)
# host = 'http://127.0.0.1:8000/sdk/'

## Local but server is demo
api_token = 'aa558b1b31012ee10e5b377ca0b1c41600ba7006'
experiment_id = '17af67f2-f7b1-4d51-98d2-88d462f91e5f'


# api_token =  sys.argv[1] # prints var1
# experiment_id = sys.argv[2] # prints var2
command = "python3 tf1_training_od.py {} {}".format(api_token, experiment_id)
host = 'https://demo.picsellia.com/sdk/'

clt = Client(api_token=api_token, host=host)
clt.exp_id = experiment_id

process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
part = "--#--Set up training"
replace_log = False
buffer = []
start_buffer = False
buffer_length = 0

while True:
    output = process.stdout.readline()
    print(output.decode("utf-8"))
    if output.decode("utf-8")  == '' and process.poll() is not None:
        break
    if output:
        if output.decode("utf-8").startswith('--#--'):
            part = output.decode("utf-8")

        if output.decode("utf-8").startswith('-----'):
            progress_line_nb = clt.line_nb
            replace_log = True

        if output.decode("utf-8").startswith('--*--'):
            replace_log = False

        if re.match("--[0-9]--", output.decode("utf-8")[:6]):
            start_buffer = True
            buffer_length = int(output.decode("utf-8")[2])

        if re.match("---[0-9]---", output.decode("utf-8")[:8]):
            start_buffer = False
            clt.send_experiment_logging(buffer, part, special='buffer')
            clt.line_nb += (len(buffer)-1)
            buffer = []

        if start_buffer:
            buffer.append(output.decode("utf-8"))
            if len(buffer)==buffer_length:
                clt.send_experiment_logging(buffer, part, special='buffer')
                clt.line_nb += (buffer_length-1)
                buffer = []
        else:
            if not replace_log:
                clt.send_experiment_logging(output.decode("utf-8"), part)
            else:
                clt.line_nb = progress_line_nb
                clt.send_experiment_logging(output.decode("utf-8"), part)
        
if buffer != []:
    clt.send_experiment_logging(buffer, part, special='buffer')
clt.send_experiment_logging(str(process.returncode), part, special='exit_code')
if process.returncode == 0 or process.returncode == "0":
    clt.update_experiment_status('succeeded')
else:
    clt.update_experiment_status('failed')
rc = process.poll()
