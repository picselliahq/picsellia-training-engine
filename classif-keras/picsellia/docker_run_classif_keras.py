import os
from picsellia import Client
from picsellia.types.enums import AnnotationFileType
import logging
import json
from classification_models.keras import Classifiers
from picsellia.types.enums import AnnotationFileType
import tensorflow as tf
from picsellia.types.enums import LogType, ExperimentStatus
from sklearn.metrics import confusion_matrix
import keras
from picsellia.exceptions import AuthenticationError
from utils import dataset as dataset_utils
os.environ['PICSELLIA_SDK_CUSTOM_LOGGING'] = "True" 
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
logging.getLogger('picsellia').setLevel(logging.INFO)

if 'api_token' not in os.environ:
    raise AuthenticationError("You must set an api_token to run this image")

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
    raise AuthenticationError("You must set the project_token or project_name and experiment_name")

experiment.start_logging_chapter('Dowloading files')

experiment.download_artifacts(with_tree=True)

artifact = experiment.get_artifact('keras-model')
model_name = artifact.filename
model_path = os.path.join(experiment.checkpoint_dir, model_name)
os.rename(os.path.join(experiment.base_dir, model_name), model_path)

dataset = experiment.list_attached_dataset_versions()[0]

annotation_path = dataset.export_annotation_file(AnnotationFileType.COCO, experiment.base_dir)
f = open(annotation_path)
annotations_dict = json.load(f)

train_assets, eval_assets, count_train, count_eval, labels = dataset.train_test_split()

labelmap = {}
for x in annotations_dict['categories']:
    labelmap[str(x['id'])] = x['name']

experiment.log('labelmap', labelmap, 'labelmap', replace=True)
experiment.log('train-split', count_train, 'bar', replace=True)
experiment.log('test-split', count_eval, 'bar', replace=True)
parameters = experiment.get_log(name='parameters').data
count_train['y'].insert(0, 0)
count_eval['y'].insert(0, 0)


train_assets.download(target_path=os.path.join(experiment.png_dir, 'train'))
eval_assets.download(target_path=os.path.join(experiment.png_dir, 'eval'))

n_classes = len(labelmap)


X_train, y_train = dataset_utils(experiment, train_assets, count_train, 'train', (parameters['image_size'], parameters['image_size']), n_classes)
X_eval, y_eval = dataset_utils(experiment, eval_assets, count_eval, 'eval', (parameters['image_size'], parameters['image_size']), n_classes)

experiment.start_logging_chapter('Init Model')

Inceptionv3, preprocess_input = Classifiers.get('inceptionv3')
X_train = preprocess_input(X_train)

base_model = Inceptionv3(input_shape=(parameters['image_size'],parameters['image_size'],3), include_top=False, weights=None)

try:
    base_model.load_weights(os.path.join(experiment.checkpoint_dir, model_name))
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(n_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
except ValueError:
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(n_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
    model.load_weights(os.path.join(experiment.checkpoint_dir, model_name))

model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(experiment.exported_model_dir, 'cp.ckpt'),
#                                                  save_weights_only=True,
#                                                  verbose=1)

experiment.start_logging_chapter('Training')

history = model.fit(X_train, y_train, epochs = int(parameters["epochs"]))

experiment.start_logging_chapter('Store model')

model.save(os.path.join(experiment.exported_model_dir, 'model.h5'))
model.save_weights(os.path.join(experiment.exported_model_dir, 'cp.ckpt'))
experiment.store(name = 'keras-model', path = os.path.join(experiment.exported_model_dir, 'model.h5'))
experiment.store(name = 'checkpoint-index', path = os.path.join(experiment.exported_model_dir, 'cp.ckpt.index'))
experiment.store(name = 'checkpoint-data', path = os.path.join(experiment.exported_model_dir, 'cp.ckpt.data-00000-of-00001'))

experiment.start_logging_chapter('Evaluation')

predictions = model.predict(X_eval)
cm=confusion_matrix(y_eval.argmax(axis = 1), predictions.argmax(axis = 1))

confusion = {
    'categories': count_train['x'],
    'values': cm.tolist()
}
experiment.log(name='confusion', data=confusion, type=LogType.HEATMAP)

for k, v in history.history.items():
    experiment.log(k, v, LogType.LINE)

experiment.update(status=ExperimentStatus.SUCCESS)