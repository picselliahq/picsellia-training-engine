import os
import sys, getopt
sys.path.append("slim")
sys.path.append(os.getcwd())
from picsellia_training.client import Client
import picsellia_training.pxl_tf as pxl_tf

import picsell_utils
import tensorflow as tf

# options_list = ["api_token=", "project_token=", "model_name=", "annotation_type=", "batch_size=",
#                  "nb_steps=", "learning_rate=", "min_score_thresh=", "num_infer=", 
#                 "incremental_or_transfer="]


# opts, args = getopt.getopt(sys.argv[1:], "", options_list)

# for opt, arg in opts:
#     if opt == "--api_token":
#         api_token = arg
#     elif opt == "--project_token":
#         project_token = arg
#     elif opt == "--model_name":
#         model_name = arg
#     elif opt == "--annotation_type":
#         annotation_type = arg
#     elif opt == "--batch_size":
#         batch_size = int(arg)
#     elif opt == "--nb_steps":
#         nb_steps = int(arg)
#     elif opt == "--learning_rate":
#         if arg == "None":
#             learning_rate = None
#     elif opt == "--min_score_thresh":
#         min_score_thresh = float(arg)
#     elif opt == "--num_infer":
#         num_infer = int(arg)
#     elif opt == "--incremental_or_transfer":
#         incremental_or_transfer = arg

api_token = os.environ['api_token']
project_token = os.environ['project_token']
model_name = os.environ['model_name']
annotation_type = os.environ['annotation_type']
batch_size = int(os.environ['batch_size'])
learning_rate = os.environ['learning_rate']
if learning_rate == "None":
    learning_rate = None
else:
    learning_rate = float(learning_rate)
min_score_thresh = float(os.environ['min_score_thresh'])
num_infer = int(os.environ['num_infer'])
incremental_or_transfer = os.environ['incremental_or_transfer']
nb_steps = int(os.environ['nb_steps'])


# api_token = 'aa558b1b31012ee10e5b377ca0b1c41600ba7006'
# project_token = 'ebf2090f-d86a-464c-b6b0-918a5389e6d5'
# model_name = 'SSD'
# annotation_type = 'rectangle'
# batch_size = 4
# learning_rate = "None"
# if learning_rate == "None":
#     learning_rate = None
# min_score_thresh = 0.5
# num_infer = 4
# incremental_or_transfer = 'incremental'
# nb_steps=200

clt = Client(api_token=api_token, host='https://demo.picsellia.com/sdk/')
clt.checkout_project(project_token=project_token)
clt.checkout_network(model_name)

clt.dl_pictures()
clt.generate_labelmap()
clt.train_test_split() 

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


picsell_utils.train(ckpt_dir=clt.checkpoint_dir, 
                     conf_dir=clt.config_dir)

metrics = picsell_utils.evaluate(clt.metrics_dir, clt.config_dir, clt.checkpoint_dir)

dict_log = picsell_utils.tfevents_to_dict(path=clt.checkpoint_dir)              

picsell_utils.export_infer_graph(ckpt_dir=clt.checkpoint_dir, 
                       exported_model_dir=clt.exported_model_dir, 
                       pipeline_config_path=clt.config_dir)


picsell_utils.infer(clt.record_dir, exported_model_dir=clt.exported_model_dir, 
      label_map_path=clt.label_path, results_dir=clt.results_dir, min_score_thresh=min_score_thresh, num_infer=num_infer, from_tfrecords=True, disp=False)


clt.send_results()
clt.send_model()
clt.send_logs(dict_log)
clt.send_metrics(metrics)
clt.send_labelmap()
