from picsellia_training.clientv2 import Client
from picsellia_tf2 import pxl_utils
from picsellia_tf2 import pxl_tf
import os
api_token = '60ef4a0b3ab8086709e2a7246cc0d134beb09d8b'
project_token = '265e8fd5-0f26-4ca2-b4c7-5f248a59f560'

experiment = Client.Experiment(api_token=api_token, project_token=project_token, host='http://127.0.0.1:8000/sdk/v2/')

exp = experiment.checkout('model 1')
print(exp)

experiment.dl_annotations()
experiment.dl_pictures()
experiment.generate_labelmap()
experiment.train_test_split()
parameters = {
    "annotation_type": 'rectangle',
    "num_steps": 1000,
    "batch_size": 4,
    "learning_rate": 1e-3
}
experiment.log('parameters', parameters)
pxl_utils.create_record_files(
        dict_annotations=experiment.dict_annotations, 
        train_list=experiment.train_list, 
        train_list_id=experiment.train_list_id, 
        eval_list=experiment.eval_list, 
        eval_list_id=experiment.eval_list_id,
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
        num_steps=parameters['num_steps'],
        batch_size=parameters['batch_size'],
        learning_rate=parameters['learning_rate'],
        annotation_type=parameters['annotation_type'],
        eval_number = 5,
        )

pxl_utils.train(
        ckpt_dir=experiment.checkpoint_dir, 
        config_dir=experiment.config_dir
    )

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
pxl_utils.infer(
    experiment.record_dir, 
    exported_model_dir=experiment.exported_model_dir, 
    label_map_path=os.path.join(experiment.base_dir,'label_map.pbtxt'), 
    results_dir=experiment.results_dir, 
    from_tfrecords=True, 
    disp=False
    )

metrics = pxl_utils.tf_events_to_dict('my_new_model2/metrics', 'eval')
logs = pxl_utils.tf_events_to_dict('my_new_model2/checkpoint', 'train')
print(logs)
experiment.store('model-latest', 'my_new_model2/exported_model.zip')
experiment.store('config', 'my_new_model2/config/pipeline.config')
experiment.store('checkpoint-data-latest', 'my_new_model2/checkpoint/ckpt-10.data-00000-of-00001')
experiment.store('checkpoint-index-latest', 'my_new_model2/checkpoint/ckpt-11.index')
experiment.log('logs', logs)
experiment.log('metrics', metrics)
