print("--#--Set up training")

import os
import sys, getopt
from picsellia_training.client import Client
import picsellia_tf2
import picsellia_tf2.pxl_tf as pxl_tf
import picsellia_tf2.pxl_utils as picsell_utils
import tensorflow as tf

api_token = 'a27d77195a9c386630e5715f7e1648936ad1ceb2'
project_token = '27b6a9b7-c1b2-46db-88d0-2b64f2857dc3'
model_name = 'mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8'.replace('_','-')
min_score_thresh = 0.5
num_infer = 4
incremental_or_transfer = 'incremental'

clt = Client(api_token=api_token, host="https://app.picsellia.com/sdk/", interactive=False)
annotation_type = 'polygon'
batch_size = 1
learning_rate = 1e-3

nb_steps = 100

clt.checkout_project(project_token=project_token)
clt.checkout_network(model_name)

clt.dl_pictures()
clt.generate_labelmap()
clt.train_test_split() 

# print("--#--Create record files")
# picsell_utils.create_record_files(dict_annotations=clt.dict_annotations, train_list=clt.train_list, 
#         train_list_id=clt.train_list_id, eval_list=clt.eval_list, 
#         eval_list_id=clt.eval_list_id,label_path=clt.label_path, 
#         record_dir=clt.record_dir, tfExample_generator=pxl_tf.tf_vars_generator, 
#         annotation_type=annotation_type
#         )

# picsell_utils.edit_config(model_selected=clt.model_selected, 
#             input_config_dir=clt.model_selected,
#             output_config_dir=clt.config_dir,
#             record_dir=clt.record_dir, 
#             label_map_path=clt.label_path, 
#             num_steps=nb_steps,
#             batch_size=batch_size,
#             learning_rate=learning_rate,
#             annotation_type=annotation_type,
#             eval_number = len(clt.eval_list),
#             incremental_or_transfer=incremental_or_transfer)

# print("--#--Start training")
# print("--5--")

# picsell_utils.train(ckpt_dir=clt.checkpoint_dir, 
#                     config=clt.config_dir)
print("---5---")

print("--#--Start export")
print("--9--")

picsell_utils.evaluate(clt.metrics_dir, clt.config_dir, clt.checkpoint_dir)        
metrics = picsell_utils.tf_events_to_dict(clt.metrics_dir, 'eval')
dict_log = picsell_utils.tf_events_to_dict(clt.checkpoint_dir, 'train')

picsell_utils.export_graph(ckpt_dir=clt.checkpoint_dir, 
                       exported_model_dir=clt.exported_model_dir, 
                       pipeline_config_dir=clt.config_dir)
print("---9---")

print("--#--Start evaluation")

picsell_utils.infer(clt.record_dir, exported_model_dir=clt.exported_model_dir, 
     label_map_path=clt.label_path, results_dir=clt.results_dir, min_score_thresh=0.01, num_infer=num_infer, from_tfrecords=True, disp=False)

print("--#--Sending to Picsell.ia")

# clt.send_results()
# clt.send_model()
# clt.send_logs(dict_log)
# clt.send_metrics(metrics)
# clt.send_labelmap()
