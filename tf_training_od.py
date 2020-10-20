print("--#--Set up training")

import os
import sys, getopt
sys.path.append("slim")
sys.path.append(os.getcwd())
from picsellia_training.client import Client
import picsellia_training.pxl_tf as pxl_tf

import picsell_utils
import tensorflow as tf

### Docker
# api_token = os.environ['api_token']
# experiment_id = os.environ['experiment_id']
# command = "python3 tf_training_od.py"
# host = 'https://demo.picsellia.com/sdk/'

### Local 
# api_token = "4d388e237d10b8a19a93517ffbe7ea32ee7f4787"
# experiment_id = 'a31a61c4-cde9-4a20-b030-3f257a2de36d'
# command = "python tf_training_od.py"
# host = 'http://127.0.0.1:8000/sdk/'

## Local but server is demo
api_token = 'aa558b1b31012ee10e5b377ca0b1c41600ba7006'
experiment_id = '5d2c6b2b-fc83-473d-a835-101eface24a2'
command = "python tf_training_od.py"
host = 'https://demo.picsellia.com/sdk/'

min_score_thresh = 0.5
num_infer = 4
incremental_or_transfer = 'incremental'

clt = Client(api_token=api_token, host=host, interactive=False)
project_token, parameters = clt.fetch_experiment_parameters(experiment_id)

model_name = clt.exp_name
annotation_type = parameters['annotation_type']
batch_size = int(parameters['batch_size'])
learning_rate = parameters['learning_rate']
if learning_rate == "None":
    learning_rate = None
else:
    learning_rate = float(learning_rate)
# min_score_thresh = float(os.environ['min_score_thresh'])
# num_infer = int(os.environ['num_infer'])
# incremental_or_transfer = os.environ['incremental_or_transfer']
nb_steps = int(parameters['epochs'])

clt.checkout_project(project_token=project_token)
clt.checkout_network(model_name)

clt.dl_pictures()
clt.generate_labelmap()
clt.train_test_split() 
print("--#--Create record files")
picsell_utils.create_record_files(dict_annotations=clt.dict_annotations, train_list=clt.train_list, 
        train_list_id=clt.train_list_id, eval_list=clt.eval_list, 
        eval_list_id=clt.eval_list_id,label_path=clt.label_path, 
        record_dir=clt.record_dir, tfExample_generator=pxl_tf.tf_vars_generator, 
        annotation_type=annotation_type
        )

picsell_utils.edit_config(model_selected=clt.model_selected, 
            config_output_dir=clt.config_dir,
            record_dir=clt.record_dir, 
            label_map_path=clt.label_path, 
            num_steps=nb_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            annotation_type=annotation_type,
            eval_number = len(clt.eval_list),
            incremental_or_transfer=incremental_or_transfer)

print("--#--Start training")
print("--5--")

picsell_utils.train(ckpt_dir=clt.checkpoint_dir, 
                     conf_dir=clt.config_dir)
print("---5---")

print("--#--Start export")
print("--9--")

metrics = picsell_utils.evaluate(clt.metrics_dir, clt.config_dir, clt.checkpoint_dir)

dict_log = picsell_utils.tfevents_to_dict(path=clt.checkpoint_dir)              

picsell_utils.export_infer_graph(ckpt_dir=clt.checkpoint_dir, 
                       exported_model_dir=clt.exported_model_dir, 
                       pipeline_config_path=clt.config_dir)
print("---9---")

print("--#--Start evaluation")

picsell_utils.infer(clt.record_dir, exported_model_dir=clt.exported_model_dir, 
      label_map_path=clt.label_path, results_dir=clt.results_dir, min_score_thresh=min_score_thresh, num_infer=num_infer, from_tfrecords=True, disp=False)

print("--#--Sending to Picsell.ia")

clt.send_results()
# clt.send_model()
# clt.send_logs(dict_log)
# clt.send_metrics(metrics)
# clt.send_labelmap()
