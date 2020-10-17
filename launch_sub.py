import subprocess
import shlex
from picsellia_training.client import Client
import os
os.environ["PYTHONUNBUFFERED"] = "1"
api_token = "4d388e237d10b8a19a93517ffbe7ea32ee7f4787"
experiment_id = 'a31a61c4-cde9-4a20-b030-3f257a2de36d'
clt = Client(api_token)
clt.exp_id = experiment_id
command = "python tf_training_od.py"
process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
part = "--#--Set up training"
replace_log = False
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
        if not replace_log:
            clt.send_experiment_logging(output.decode("utf-8"), part)
        else:
            clt.line_nb = progress_line_nb
            clt.send_experiment_logging(output.decode("utf-8"), part)
        
    
rc = process.poll()