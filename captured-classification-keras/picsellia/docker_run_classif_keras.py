import os
import random
import numpy as np
from picsellia.types.enums import AnnotationFileType
import logging
import json
from classification_models.keras import Classifiers
from picsellia.types.enums import AnnotationFileType
import tensorflow as tf
from picsellia.types.enums import LogType
from picsellia.sdk.asset import MultiAsset
from picsellia.sdk.experiment import Experiment
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
import utils
from utils import(
    _move_files_in_class_directories, get_experiment, get_train_test_valid_datasets_from_experiment
) 
from pycocotools.coco import COCO

os.environ['PICSELLIA_SDK_CUSTOM_LOGGING'] = "True" 
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
# os.environ["PICSELLIA_SDK_SECTION_HANDLER"] = "1"

logging.getLogger('picsellia').setLevel(logging.INFO)


experiment: Experiment = get_experiment()

model_architecture = os.environ["architecture"]

experiment.start_logging_chapter('Dowloading files')

experiment.download_artifacts(with_tree=True)

artifact = experiment.get_artifact('keras-model')
model_name = artifact.filename
model_path = os.path.join(experiment.checkpoint_dir, model_name)

os.rename(os.path.join(experiment.base_dir, model_name), model_path)

is_split, train_ds, test_ds, valid_ds = get_train_test_valid_datasets_from_experiment(experiment)
# if not is_split:
dataset = train_ds


print("Downloading annotation COCO file ...")
annotation_path = dataset.export_annotation_file(AnnotationFileType.COCO, experiment.base_dir)
print("Downloading annotation COCO file ... OK")

coco = COCO(annotation_path)
labelmap = {}
for x in coco.cats:
    labelmap[str(x)] = coco.cats[x]['name']

n_classes = len(labelmap)

train, val, count_train, count_eval, labels = dataset.train_test_split(prop = 0.7)

train_list = train.items
test_list = val.items
random.shuffle(train_list)
random.shuffle(test_list)
train_assets = MultiAsset(dataset.connexion, dataset.id, train_list)
eval_assets = MultiAsset(dataset.connexion, dataset.id, test_list)

train_assets.download(target_path=os.path.join(experiment.png_dir, 'train'))
eval_assets.download(target_path=os.path.join(experiment.png_dir, 'eval'))

_move_files_in_class_directories(coco=coco, base_imdir=os.path.join(experiment.png_dir, 'train'))
_move_files_in_class_directories(coco=coco, base_imdir=os.path.join(experiment.png_dir, 'eval'))

experiment.log('labelmap', labelmap, 'labelmap', replace=True)
experiment.log('train-split', count_train, 'bar', replace=True)
experiment.log('test-split', count_eval, 'bar', replace=True)
count_train['y'].insert(0, 0)
count_eval['y'].insert(0, 0)

parameters = experiment.get_log(name='parameters').data
random_seed=parameters.get("random_seed", 12)
target_width = int(parameters.get("resized_width", 200))
target_height = int(parameters.get("resized_height", 200))
EPOCHS=parameters.get("epochs", 100)
INITIAL_LR=parameters.get("learning_rate", 0.0005)
target_size = (target_width, target_height)
target_input = (target_width, target_height, 3)
batch_size = int(parameters.get("batch_size", 128))
target_size = (target_width, target_height)

train_datagen = ImageDataGenerator(rescale=1./255.,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    horizontal_flip=True,
    brightness_range=[0.7,1.3],
    dtype='float32'
    )

test_datagen = ImageDataGenerator(rescale=1./255.,dtype='float32')

train_generator = train_datagen.flow_from_directory(
        os.path.join(experiment.png_dir, 'train'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        seed=random_seed,
        shuffle=True
    )
test_generator = test_datagen.flow_from_directory(
    os.path.join(experiment.png_dir, 'eval'),
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    seed=random_seed,
    shuffle=True
)

experiment.start_logging_chapter('Init Model')


Architecture, preprocess_input = Classifiers.get(model_architecture)


base_model = Architecture(input_shape=target_input, include_top=False, weights=None)

try:
    base_model.load_weights(os.path.join(experiment.checkpoint_dir, model_name))
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(n_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
except ValueError as e:
    print(str(e))
    print("Adding an extra Average Pooling Layer .. OK")
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(n_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
    model.load_weights(os.path.join(experiment.checkpoint_dir, model_name))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=parameters['learning_rate']), loss='categorical_crossentropy', metrics=['accuracy', utils.f1_micro])

experiment.start_logging_chapter('Training')

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
history = model.fit(train_generator, batch_size = batch_size, epochs = EPOCHS, validation_data=test_generator, callbacks=[utils.Metrics(val_data=test_generator, batch_size=batch_size, experiment=experiment)], verbose=1, class_weight=dict(enumerate(class_weights)))
for k, v in history.history.items():
    try:
        experiment.log(k, v, LogType.LINE)
    except Exception as e:
        print(f"can't send {v}")

model.save(os.path.join(experiment.exported_model_dir, 'model.h5'))
model.save_weights(os.path.join(experiment.exported_model_dir, 'cp.ckpt'))
tf.saved_model.save(model, os.path.join(experiment.exported_model_dir, 'saved_model'))
experiment.store("model-latest", os.path.join(experiment.exported_model_dir, 'saved_model'), do_zip=True)
experiment.store(name = 'keras-model', path = os.path.join(experiment.exported_model_dir, 'model.h5'))
experiment.store(name = 'checkpoint-index', path = os.path.join(experiment.exported_model_dir, 'cp.ckpt.index'))
experiment.store(name = 'checkpoint-data', path = os.path.join(experiment.exported_model_dir, 'cp.ckpt.data-00000-of-00001'))

experiment.start_logging_chapter('Evaluation')

predictions = model.predict(test_generator)


test_accuracy = accuracy_score(test_generator.classes, predictions.argmax(axis = 1))

experiment.log(name='test_accuracy', data=test_accuracy.item(), type=LogType.VALUE)

cm=confusion_matrix(test_generator.classes, predictions.argmax(axis = 1))

confusion = {
    'categories': count_train['x'],
    'values': cm.tolist()
}
log = experiment.log(name='confusion', data=confusion, type=LogType.HEATMAP)