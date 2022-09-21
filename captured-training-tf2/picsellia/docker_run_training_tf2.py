
import os
from picsellia import Client
from picsellia.exceptions import AuthenticationError
from picsellia_tf2 import pxl_utils
from picsellia_tf2 import pxl_tf
import logging
os.environ['PICSELLIA_SDK_CUSTOM_LOGGING'] = "True" 
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
logging.getLogger('picsellia').setLevel(logging.INFO)

if 'api_token' not in os.environ:
    raise AuthenticationError("You must set an api_token to run this image")

api_token = os.environ["api_token"]

if "host" not in os.environ:
    host = "https://app.picsellia.com/sdk/v1"
else:
    host = os.environ["host"]

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
    experiment = project.get_experiment(experiment_name, tree=True, with_artifacts=True)
else:
    raise AuthenticationError("You must set the project_token or project_name and experiment_name")

experiment.start_logging_chapter('Downloading Data')

experiment.download_annotations()
experiment.download_pictures()
experiment.generate_labelmap()
experiment.log('labelmap', experiment.label_map, 'labelmap', replace=True)
train_list, eval_list, train_list_id, eval_list_id, label_train, label_eval, categories = experiment.train_test_split()

train_split = {
    'x': categories,
    'y': label_train,
    'image_list': train_list_id
}
experiment.log('train-split', train_split, 'bar', replace=True)

test_split = {
    'x': categories,
    'y': label_eval,
    'image_list': eval_list_id
}
experiment.log('test-split', test_split, 'bar', replace=True)
parameters = experiment.get_log(name='parameters')

experiment.start_logging_chapter('Create records')

pxl_utils.create_record_files(
        dict_annotations=experiment.dict_annotations, 
        train_list=train_list, 
        train_list_id=train_list_id, 
        eval_list=eval_list, 
        eval_list_id=eval_list_id,
        label_path=experiment.label_path, 
        record_dir=experiment.record_dir, 
        tfExample_generator=pxl_tf.tf_vars_generator, 
        annotation_type=parameters['annotation_type']
        )
    
pxl_utils.edit_config(
        model_selected=experiment.checkpoint_dir, 
        input_config_dir=experiment.config_dir,
        output_config_dir=experiment.config_dir,
        record_dir=experiment.record_dir, 
        label_map_path=experiment.label_path, 
        num_steps=parameters["steps"],
        batch_size=parameters['batch_size'],
        learning_rate=parameters['learning_rate'],
        annotation_type=parameters['annotation_type'],
        eval_number = 5,
        parameters=parameters,
        )

experiment.start_logging_chapter('Start training')

pxl_utils.train(
        ckpt_dir=experiment.checkpoint_dir, 
        config_dir=experiment.config_dir,
        log_real_time=experiment,
    )

experiment.start_logging_chapter('Start evaluation')

experiment.start_logging_buffer(9)


pxl_utils.evaluate(
    experiment.metrics_dir, 
    experiment.config_dir, 
    experiment.checkpoint_dir
    )        
pxl_utils.export_graph(
    ckpt_dir=experiment.checkpoint_dir, 
    exported_model_dir=experiment.exported_model_dir, 
    config_dir=experiment.config_dir
    )
experiment.end_logging_buffer()
experiment.start_logging_chapter('Store artifacts')

experiment.store('model-latest')
experiment.store('config')
experiment.store('checkpoint-data-latest')
experiment.store('checkpoint-index-latest')


experiment.start_logging_chapter('Send logs')

metrics = pxl_utils.tf_events_to_dict('{}/metrics'.format(experiment.name), 'eval')
logs = pxl_utils.tf_events_to_dict('{}/checkpoint'.format(experiment.name), 'train')

for variable in logs.keys():
    data = {
        'steps': logs[variable]["steps"],
        'values': logs[variable]["values"]
    }
    experiment.log('-'.join(variable.split('/')), data, 'line', replace=True)
    
experiment.log('metrics', metrics, 'table', replace=True)

experiment.start_logging_chapter('Compute Confusion matrix')

conf, eval = pxl_utils.get_confusion_matrix(
    input_tfrecord_path=os.path.join(experiment.record_dir, 'eval.record'),
    model=os.path.join(experiment.exported_model_dir, 'saved_model'),
    labelmap=experiment.label_map
)


confusion = {
    'categories': list(experiment.label_map.values()),
    'values': conf.tolist()
}

experiment.log('confusion-matrix', confusion, 'heatmap', replace=True)
experiment.log('evaluation', eval, 'evaluation', replace=True)


experiment.start_logging_chapter('Start inference')


pxl_utils.infer(
    experiment.record_dir, 
    exported_model_dir=experiment.exported_model_dir, 
    label_map_path=os.path.join(experiment.base_dir,'label_map.pbtxt'), 
    results_dir=experiment.results_dir, 
    from_tfrecords=True, 
    disp=False
)


