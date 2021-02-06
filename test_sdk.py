import os
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from picsellia_training.clientv2 import Client
from picsellia_tf2 import pxl_utils
from picsellia_tf2 import pxl_tf


api_token = 'f54f181b21a6f359bee533ec00381e3c7c2ce4e9'
project_token = 'b27381b0-0c29-49f7-842a-f5116e61653b'

experiment = Client.Experiment(api_token=api_token, project_token=project_token, host='https://beta.picsellia.com/sdk/v2/')

exp = experiment.checkout('detection-v2', tree=True)

experiment.dl_annotations()
experiment.dl_pictures()
experiment.generate_labelmap()
experiment.train_test_split()
train_split = {
    'x': experiment.categories,
    'y': experiment.train_repartition,
    'image_list': experiment.train_list_id
}
experiment.log('train-split', train_split, 'bar', replace=True)

train_split = {
    'x': experiment.categories,
    'y': experiment.test_repartition,
    'image_list': experiment.eval_list_id
}
experiment.log('test-split', train_split, 'bar', replace=True)
# parameters = experiment.get_data(name='parameters')[0]["data"]
# experiment.log('parameters', parameters)
# pxl_utils.create_record_files(
#         dict_annotations=experiment.dict_annotations, 
#         train_list=experiment.train_list, 
#         train_list_id=experiment.train_list_id, 
#         eval_list=experiment.eval_list, 
#         eval_list_id=experiment.eval_list_id,
#         label_path=experiment.label_path, 
#         record_dir=experiment.record_dir, 
#         tfExample_generator=pxl_tf.tf_vars_generator, 
#         annotation_type=parameters['annotation_type']
#         )
    
# pxl_utils.edit_config(
#         model_selected=experiment.checkpoint_dir, 
#         input_config_dir=experiment.config_dir,
#         output_config_dir=experiment.config_dir,
#         record_dir=experiment.record_dir, 
#         label_map_path=experiment.label_path, 
#         num_steps=1000,
#         batch_size=parameters['batch_size'],
#         learning_rate=parameters['learning_rate'],
#         annotation_type=parameters['annotation_type'],
#         eval_number = 5,
#         )

# pxl_utils.train(
#         ckpt_dir=experiment.checkpoint_dir, 
#         config_dir=experiment.config_dir
#     )

# pxl_utils.evaluate(
#     experiment.metrics_dir, 
#     experiment.config_dir, 
#     experiment.checkpoint_dir
#     )        
# pxl_utils.export_graph(
#     ckpt_dir=experiment.checkpoint_dir, 
#     exported_model_dir=experiment.exported_model_dir, 
#     config_dir=experiment.config_dir
#     )
# pxl_utils.infer(
#     experiment.record_dir, 
#     exported_model_dir=experiment.exported_model_dir, 
#     label_map_path=os.path.join(experiment.base_dir,'label_map.pbtxt'), 
#     results_dir=experiment.results_dir, 
#     from_tfrecords=True, 
#     disp=False
#     )

# metrics = pxl_utils.tf_events_to_dict('{}/metrics'.format(exp.experiment_name), 'eval')
# logs = pxl_utils.tf_events_to_dict('{}/checkpoint'.format(exp.experiment_name), 'train')
# # # experiment.store('model-latest', 'my_new_model2/exported_model.zip')
# # # experiment.store('config', 'my_new_model2/config/pipeline.config')
# # # experiment.store('checkpoint-data-latest', 'my_new_model2/checkpoint/ckpt-10.data-00000-of-00001')
# # # experiment.store('checkpoint-index-latest', 'my_new_model2/checkpoint/ckpt-11.index')
# for variable in logs.keys():
#     data = {
#         'steps': logs[variable]["steps"],
#         'values': logs[variable]["values"]
#     }
#     experiment.log('-'.join(variable.split('/')), data, 'line', replace=True)
# experiment.log('metrics', metrics, 'table', replace=True)
