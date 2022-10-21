import os
from picsellia import Client
from picsellia.types.enums import AnnotationFileType
from picsellia.exceptions import AuthenticationError
from picsellia_tf2 import pxl_utils
from picsellia_tf2 import pxl_tf
import logging
import json

from classification_models.keras import Classifiers
import json
from picsellia.client import Client
from picsellia.types.enums import InferenceType
import os
from picsellia.types.enums import Framework
from picsellia.types.enums import AnnotationFileType
from picsellia.sdk.asset import Asset, MultiAsset
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from picsellia.types.enums import LogType
import keras
from sklearn.metrics import confusion_matrix

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

experiment.download_artifacts(with_tree=True)

dataset = experiment.list_attached_dataset_versions()[0]
dataset.download(experiment.png_dir)
annotation_path = dataset.export_annotation_file(AnnotationFileType.COCO, experiment.base_dir)
f = open(annotation_path)
annotations_dict = json.load(f)
label_path = pxl_utils.generate_label_map(
    classes=[x['name'] for x in annotations_dict['categories']],
    output_path=experiment.base_dir,
)

train_assets, eval_assets, count_train, count_eval, labels = dataset.train_test_split()

labelmap = {}
for x in annotations_dict['categories']:
    labelmap[x['id']] = x['name']

experiment.log('labelmap', labelmap, 'labelmap', replace=True)
experiment.log('train-split', count_train, 'bar', replace=True)

experiment.log('test-split', count_eval, 'bar', replace=True)
parameters = experiment.get_log(name='parameters').data

experiment.start_logging_chapter('Create records')
x = lambda x : os.path.join(experiment.png_dir, x)

pxl_utils.create_record_files(
        annotation_path=annotation_path,
        train_list=[x(e.filename) for e in train_assets],
        train_list_id=[e.id for e in train_assets],
        eval_list=[x(e.filename) for e in eval_assets],
        eval_list_id=[e.id for e in eval_assets],
        label_path=label_path,
        record_dir=experiment.record_dir,
        tfExample_generator=pxl_tf.tf_vars_generator,
        annotation_type=parameters['annotation_type']
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

count_train['y'].insert(0, 0)
count_eval['y'].insert(0, 0)
train_assets = MultiAsset(dataset.connexion, dataset.id, train_assets)
eval_assets = MultiAsset(dataset.connexion, dataset.id, eval_assets)

image_path = './MAT_images/'

train_assets.download(target_path=os.path.join(experiment.img_dir, 'train'))
eval_assets.download(target_path=os.path.join(experiment.img_dir, 'eval'))

n_classes = len(labelmap)

def dataset(assets, count, dataset_type, new_size):

    X = []
    for i in range(len(assets)):
        path = image_path + dataset_type + '/' + assets[i].filename
        x = Image.open(path).convert('RGB')
        x = x.resize(new_size)
        x = np.array(x)
        X.append(x)

    y = np.zeros(len(assets))
    indices = np.cumsum(count['y'])
    for i in range(len(indices)-1):
        y[indices[i]:indices[i+1]] = np.ones(len(y[indices[i]:indices[i+1]]))*(i)
    y = to_categorical(y, n_classes)

    return np.asarray(X), y

X_train, y_train = dataset(train_assets, count_train, 'train', (parameters['image_size'], parameters['image_size']))
X_eval, y_eval = dataset(eval_assets, count_eval, 'eval', (parameters['image_size'], parameters['image_size']))

experiment.log(name='parameters', data=parameters, type=LogType.TABLE)

ResNet18, preprocess_input = Classifiers.get('resnet18')

X_train = preprocess_input(X_train)

# build model

# base_model = create_model()

# base_model.load_weights(checkpoint_path)


base_model = ResNet18(input_shape=(parameters['image_size'],parameters['image_size'],3), weights='imagenet', include_top=False)
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])

# train
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(X_train, y_train, epochs = parameters['epochs'], callbacks=[cp_callback])

experiment.store(name = 'checkpoint', path = checkpoint_path + '/' + 'checkpoint')
experiment.store(name = 'cp.ckpt.index', path = checkpoint_path + '/' + 'cp.ckpt.index')
experiment.store(name = 'cp.ckpt.data-00000-of-00001', path = checkpoint_path + '/' + 'cp.ckpt.data-00000-of-00001')

predictions = model.evaluate(X_eval)

cm=confusion_matrix(y_eval.argmax(axis = 1), predictions.argmax(axis = 1))

confusion = {
    'categories': count_train['x'],
    'values': cm.tolist()
}
experiment.log(name='confusion', data=confusion, type=LogType.HEATMAP)

for k, v in history.history.items():
    experiment.log(k, v, LogType.LINE)