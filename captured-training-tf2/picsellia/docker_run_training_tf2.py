
import os
from picsellia import Client
from picsellia import utils
from picsellia.types.enums import AnnotationFileType
from evaluator import Evaluator
from picsellia.exceptions import ResourceNotFoundError

from picsellia_tf2 import pxl_utils
from picsellia_tf2 import pxl_tf

import logging
import json
import shutil

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
parameters = experiment.get_log(name='parameters').data

attached_datasets = experiment.list_attached_dataset_versions()
if len(attached_datasets) == 3:
    try:
        train_ds = experiment.get_dataset(name="train")
    except Exception:
        raise ResourceNotFoundError("Found 3 attached datasets, but can't find any 'train' dataset.\n \
                                            expecting 'train', 'test', ('val' or 'eval')")
    try:
        test_ds = experiment.get_dataset(name="test")
    except Exception:
        raise ResourceNotFoundError("Found 3 attached datasets, but can't find any 'test' dataset.\n \
                                            expecting 'train', 'test', ('val' or 'eval')")
    try:
        eval_ds = experiment.get_dataset(name="val")
    except Exception:
        try:
            eval_ds = experiment.get_dataset(name="eval")
        except Exception:
            raise ResourceNotFoundError("Found 3 attached datasets, but can't find any 'eval' dataset.\n \
                                                expecting 'train', 'test', ('val' or 'eval')")


    labels = train_ds.list_labels()
    label_names = [label.name for label in labels]
    labelmap = {str(i): label.name for i, label in enumerate(labels)}
    label_path = pxl_utils.generate_label_map(
        classes=label_names,
        output_path=experiment.base_dir,
    )
    
    train_ds.download(
            target_path=os.path.join(experiment.png_dir, 'train'), max_workers=8
        )
    train_split = {'x': list(pxl_utils.retrieve_stats(train_ds)['label_repartition'].keys()), 'y': list(pxl_utils.retrieve_stats(train_ds)['label_repartition'].values())}
    # train_annotation_path = train_ds.export_annotation_file(AnnotationFileType.COCO, experiment.base_dir)
    # with open(train_annotation_path, "r") as f:
    #     train_annotations = json.load(f)
        
    train_annotation_path = train_ds.build_coco_file_locally(enforced_ordered_categories=label_names)
    train_annotations = train_annotation_path.dict()
    categories_dict = [category['name'] for category in train_annotations['categories']]
    for label in label_names:
        if label not in categories_dict:
            train_annotations['categories'].append({"id": len(train_annotations['categories']), "name": label, "supercategory": ""})
            
        
    train_annotations, _ = pxl_utils.format_coco_file(
        imgdir=experiment.png_dir,
        annotations=train_annotations,
        train_assets=train_ds.list_assets(),
        test_assets=[]
    )
    
    eval_ds.download(
            target_path=os.path.join(experiment.png_dir, 'eval'), max_workers=8
        )
    eval_split = {'x': list(pxl_utils.retrieve_stats(eval_ds)['label_repartition'].keys()), 'y': list(pxl_utils.retrieve_stats(eval_ds)['label_repartition'].values())}

    eval_annotation_path = eval_ds.build_coco_file_locally(enforced_ordered_categories=label_names)
    eval_annotations = eval_annotation_path.dict()
    categories_dict = [category['name'] for category in eval_annotations['categories']]
    for label in label_names:
        if label not in categories_dict:
            eval_annotations['categories'].append({"id": len(eval_annotations['categories']), "name": label, "supercategory": ""})
        
    _, eval_annotations = pxl_utils.format_coco_file(
        imgdir=experiment.png_dir,
        annotations=eval_annotations,
        train_assets=[],
        test_assets=eval_ds.list_assets()
    )
    
    test_ds.download(
            target_path=os.path.join(experiment.png_dir, 'test'), max_workers=8
        )
    test_split = {'x': list(pxl_utils.retrieve_stats(test_ds)['label_repartition'].keys()), 'y': list(pxl_utils.retrieve_stats(test_ds)['label_repartition'].values())}

    test_annotation_path = test_ds.build_coco_file_locally(enforced_ordered_categories=label_names)
    test_annotations = test_annotation_path.dict()
    categories_dict = [category['name'] for category in test_annotations['categories']]
    for label in label_names:
        if label not in categories_dict:
            test_annotations['categories'].append({"id": len(test_annotations['categories']), "name": label, "supercategory": ""})
        
    _, test_annotations = pxl_utils.format_coco_file(
        imgdir=experiment.png_dir,
        annotations=test_annotations,
        train_assets=[],
        test_assets=test_ds.list_assets()
    )
    
else:
    dataset = experiment.list_attached_dataset_versions()[0]
    
    labels = dataset.list_labels()
    label_names = [label.name for label in labels]
    labelmap = {str(i): label.name for i, label in enumerate(labels)}
    label_path = pxl_utils.generate_label_map(
        classes=label_names,
        output_path=experiment.base_dir,
    )
    
    train_assets, test_assets, train_split, test_split, categories = dataset.train_test_split()
    train_assets.download(target_path=os.path.join(experiment.png_dir, 'train'), max_workers=8
        )
    test_assets.download(target_path=os.path.join(experiment.png_dir, 'test'), max_workers=8
        )
        
    annotation_path = dataset.build_coco_file_locally(enforced_ordered_categories=label_names)
    annotations = annotation_path.dict()
    categories_dict = [category['name'] for category in annotations['categories']]
    for label in label_names:
        if label not in categories_dict:
            annotations['categories'].append({"id": len(annotations['categories']), "name": label, "supercategory": ""})
            
    train_annotations, test_annotations = pxl_utils.format_coco_file(
        imgdir=experiment.png_dir,
        annotations=annotations,
        train_assets=train_assets,
        test_assets=test_assets
    )

experiment.log('labelmap', labelmap, 'labelmap', replace=True)
experiment.log('train-split', pxl_utils.sort_split(train_split, label_names), 'bar', replace=True)
experiment.log('eval-split', pxl_utils.sort_split(eval_split, label_names), 'bar', replace=True)
experiment.log('test-split', pxl_utils.sort_split(test_split, label_names), 'bar', replace=True)

print("\n")
experiment.start_logging_chapter('Create records')
print("\n")

x = lambda x : os.path.join(experiment.png_dir, x)

pxl_utils.create_record_files(
        train_annotations=train_annotations,
        eval_annotations=eval_annotations,
        test_annotations=test_annotations,
        label_path=label_path,
        record_dir=experiment.record_dir,
        tfExample_generator=pxl_tf.tf_vars_generator,
        annotation_type=parameters['annotation_type']
        )

# edit training config
training_config_dir = experiment.config_dir
test_config = os.path.join(experiment.base_dir, 'test_config')
if os.path.isfile(os.path.join(training_config_dir, 'pipeline.config')):
    shutil.copy(os.path.join(training_config_dir, 'pipeline.config'), test_config)
    
pxl_utils.edit_config(
        model_selected=experiment.checkpoint_dir, 
        input_config_dir=training_config_dir,
        output_config_dir=training_config_dir,
        train_record_path=os.path.join(experiment.record_dir, 'train.record'),
        eval_record_path=os.path.join(experiment.record_dir, 'eval.record'),
        label_map_path=label_path, 
        num_steps=parameters["steps"],
        batch_size=parameters['batch_size'],
        learning_rate=parameters['learning_rate'],
        annotation_type=parameters['annotation_type'],
        parameters=parameters
        )

# edit final test config

pxl_utils.edit_config(
        model_selected=experiment.checkpoint_dir, 
        input_config_dir=test_config,
        output_config_dir=test_config,
        train_record_path=os.path.join(experiment.record_dir, 'train.record'),
        eval_record_path=os.path.join(experiment.record_dir, 'test.record'),
        label_map_path=label_path, 
        num_steps=parameters["steps"],
        batch_size=parameters['batch_size'],
        learning_rate=parameters['learning_rate'],
        annotation_type=parameters['annotation_type'],
        parameters=parameters
        )

print("\n")
experiment.start_logging_chapter('Start training')
print("\n")

pxl_utils.train(
        ckpt_dir=experiment.checkpoint_dir, 
        config_dir=training_config_dir,
        log_real_time=experiment,
        evaluate_fn=pxl_utils.evaluate,
        read_logs_fn=pxl_utils.tf_events_to_dict,
        checkpoint_every_n=parameters.get('checkpoint_every_n', 5)
    )

print("\n")
experiment.start_logging_chapter('Store artifacts')
print("\n")

pxl_utils.export_graph(
    ckpt_dir=experiment.checkpoint_dir, 
    exported_model_dir=experiment.exported_model_dir, 
    config_dir=training_config_dir
    )
experiment.store('model-latest')
experiment.store('config')
experiment.store('checkpoint-data-latest')
experiment.store('checkpoint-index-latest')


print("\n")
experiment.start_logging_chapter('Computing metrics on test dataset')
print("\n")

experiment.start_logging_buffer(9)

test_metrics_dir = os.path.join(experiment.base_dir, 'test_metrics')
if not os.path.exists(test_metrics_dir):
    os.makedirs(test_metrics_dir)
    
pxl_utils.evaluate(
    test_metrics_dir, 
    test_config, 
    experiment.checkpoint_dir
    )

metrics = pxl_utils.tf_events_to_dict('{}/test_metrics'.format(experiment.name), 'eval')
experiment.log('metrics', metrics, 'table', replace=True)

conf, eval = pxl_utils.get_confusion_matrix(
    input_tfrecord_path=os.path.join(experiment.record_dir, 'test.record'),
    model=os.path.join(experiment.exported_model_dir, 'saved_model'),
    labelmap=labelmap
)

confusion = {
    'categories': list(labelmap.values()),
    'values': conf.tolist()
}

experiment.log('confusion-matrix', confusion, 'heatmap', replace=True)

experiment.end_logging_buffer()

# pxl_utils.infer(
#     experiment.record_dir, 
#     exported_model_dir=experiment.exported_model_dir, 
#     label_map_path=label_path, 
#     results_dir=experiment.results_dir, 
#     from_tfrecords=True, 
#     disp=False
# )

print("\n")
experiment.start_logging_chapter('Starting Evaluation')
print("\n")

try:
    X = Evaluator(
        client=client,
        experiment=experiment, # same
        dataset=test_ds,
        asset_list=test_ds.list_assets()
    )

    X.setup_preannotation_job()
    X.preannotate()
except Exception as e:
    print(f"Error during evaluation: {e}")