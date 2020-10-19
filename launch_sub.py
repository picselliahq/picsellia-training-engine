import subprocess
import shlex
from picsellia_training.client import Client
import os
import re 
os.environ["PYTHONUNBUFFERED"] = "1"
api_token = "4d388e237d10b8a19a93517ffbe7ea32ee7f4787"
experiment_id = 'a31a61c4-cde9-4a20-b030-3f257a2de36d'
clt = Client(api_token)
clt.exp_id = experiment_id
command = "python tf_training_od.py"
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
print(process.returncode)
clt.send_experiment_logging(str(process.returncode), part, special='exit_code')
rc = process.poll()