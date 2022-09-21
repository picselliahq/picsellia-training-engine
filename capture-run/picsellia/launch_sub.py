import subprocess
import shlex
from picsellia import Client
import os
import re 
os.environ["PYTHONUNBUFFERED"] = "1"
os.chdir('picsellia')
import sys
from picsellia.exceptions import AuthenticationError

if 'api_token' not in os.environ:
    raise AuthenticationError("You must set an api_token to run this image")
api_token = os.environ["api_token"]

if "project_token" not in os.environ:
    raise AuthenticationError("You must set a valid project token to launch runs")
project_token = os.environ["project_token"]

if "run_id" not in os.environ:
    raise AuthenticationError("You must set a valid run id to launch run")
run_id = os.environ["run_id"]

if "host" not in os.environ:
    host = "https://app.picsellia.com/sdk/v2/"
else:
    host = os.environ["host"]

client = Client(api_token=api_token, host=host)
run = client.get_run_by_id(run_id)
exp = run.get_experiment()
os.environ["experiment_id"] = exp.id


filename = exp.download_script()

command = "python3 {}/{}".format(os.getcwd(), filename)
process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
part = "--#--Start Run"
replace_log = False
buffer = []
start_buffer = False
buffer_length = 0
exp.send_experiment_logging(part, part)

last_line = ""
while True:
    with open('{}-logs.txt'.format(exp.id), 'w') as f:
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
                if replace_log:
                    f.write(last_line + os.linesep)
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
            
            last_line = output.decode("utf-8")

            if not replace_log:
                f.write(output.decode("utf-8") + os.linesep)
        
if buffer != []:
    exp.send_experiment_logging(buffer, part, special='buffer')
exp.send_experiment_logging(str(process.returncode), part, special='exit_code')
exp.store('logs','{}-logs.txt'.format(exp.id))

if process.returncode == 0 or process.returncode == "0":
    exp.update(status='success')
    exp.update_run(status='success')
else:
    exp.update(status='failed')
    exp.update_run(status='failed')
exp.end_run()
rc = process.poll()