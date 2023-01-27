
import os
from picsellia import Client
from picsellia.types.enums import AnnotationFileType
from picsellia_tf2 import pxl_utils
from picsellia_tf2 import pxl_tf
import logging
os.environ['PICSELLIA_SDK_CUSTOM_LOGGING'] = "True" 
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
os.environ["PICSELLIA_SDK_SECTION_HANDLER"] = "1"
logging.getLogger('picsellia').setLevel(logging.INFO)

if 'api_token' not in os.environ:
    raise RuntimeError("You must set an api_token to run this image")

api_token = os.environ["api_token"]

if "host" not in os.environ:
    host = "https://app.picsellia.com"
else:
    host = os.environ["host"]

if "organization_id" not in os.environ:
    organization_id = None
else:
    organization_id = os.environ["organization_id"]

client = Client(
    api_token=api_token,
    host=host,
    organization_id=organization_id
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
    raise RuntimeError("You must set the project_token or project_name and experiment_name")

experiment.download_artifacts(with_tree=True)

dataset = experiment.list_attached_dataset_versions()[0]
dataset.download(experiment.png_dir)
annotation_path = dataset.export_annotation_file(AnnotationFileType.COCO, experiment.base_dir)
labels = dataset.list_labels()
label_path = pxl_utils.generate_label_map(
    classes=[e.name for e in labels],
    output_path=experiment.base_dir,
)

train_assets, eval_assets, train_split, test_split, categories = dataset.train_test_split()

labelmap = {}

for i, label in enumerate(categories):
    labelmap[str(i+1)] = label.name

experiment.log('labelmap', labelmap, 'labelmap', replace=True)
experiment.log('train-split', train_split, 'bar', replace=True)

experiment.log('test-split', test_split, 'bar', replace=True)
parameters = experiment.get_log(name='parameters').data

experiment.start_logging_chapter('Create records')
x = lambda x : os.path.join(experiment.png_dir, x)

pxl_utils.create_record_files(
        annotation_path=annotation_path,
        train_list=train_assets,
        eval_list=eval_assets,
        label_path=label_path,
        record_dir=experiment.record_dir,
        tfExample_generator=pxl_tf.tf_vars_generator,
        annotation_type=parameters['annotation_type'],
        imgdir=experiment.png_dir
        )

pxl_utils.edit_config(
        model_selected=experiment.checkpoint_dir, 
        input_config_dir=experiment.config_dir,
        output_config_dir=experiment.config_dir,
        record_dir=experiment.record_dir, 
        label_map_path=label_path, 
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
    labelmap=labelmap
)


confusion = {
    'categories': list(labelmap.values()),
    'values': conf.tolist()
}

experiment.log('confusion-matrix', confusion, 'heatmap', replace=True)


experiment.start_logging_chapter('Start inference')


pxl_utils.infer(
    experiment.record_dir, 
    exported_model_dir=experiment.exported_model_dir, 
    label_map_path=label_path, 
    results_dir=experiment.results_dir, 
    from_tfrecords=True, 
    disp=False
)
