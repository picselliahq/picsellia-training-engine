import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import models
from picsellia.client import Client
from picsellia.pxl_exceptions import AuthenticationError
from sklearn.metrics import classification_report, confusion_matrix


if 'api_token' not in os.environ:
    raise AuthenticationError("You must set an api_token to run this image")

api_token = os.environ["api_token"]

if "host" not in os.environ:
    host = "https://app.picsellia.com/sdk/v2/"
else:
    host = os.environ["host"]

if "experiment_id" in os.environ:
    experiment_id = os.environ['experiment_id']
    project_token = os.environ['project_token']
    experiment = Client.Experiment(api_token=api_token, project_token=project_token, host=host, interactive=False)
    experiment.id = experiment_id
    exp = experiment.checkout(id=experiment_id, with_file=True)
else:
    if "experiment_name" in os.environ and "project_token" in os.environ:
        project_token = os.environ['project_token']
        experiment_name = os.environ['experiment_name']
        experiment = Client.Experiment(api_token=api_token, project_token=project_token, host=host, interactive=False)
        exp = experiment.checkout(experiment_name, with_file=True)
    else:
        raise AuthenticationError("You must either set the experiment id or the project token + experiment_name")

exp.dl_annotations()
exp.dl_pictures()
exp.train_test_split()
train_split = {
    'x': exp.categories,
    'y': exp.train_repartition,
    'image_list': exp.train_list_id
}
exp.log('train-split', train_split, 'bar', replace=True)

test_split = {
    'x': exp.categories,
    'y': exp.test_repartition,
    'image_list': exp.eval_list_id
}
exp.log('test-split', test_split, 'bar', replace=True)

labels_index = [e for e in range(1, len(exp.categories)+1)]
labelmap = dict(zip(labels_index, exp.categories))
exp.log('labelmap', labelmap, 'labelmap', replace=True)

splits = ["train", "validation"]
for split in splits:
    if not split in os.listdir("images"):
        os.mkdir(f"images/{split}")
    for category in exp.categories:
        if not category in os.listdir(f"images/{split}"):
            os.mkdir(f"images/{split}/{category}")

train_dir = "images/train"
validation_dir = "images/validation"

for train_img in exp.train_list:
    filename = train_img.split("/")[-1]
    label = filename.split("_")[0]
    os.rename(train_img, f"{train_dir}/{label}/{filename}")

for eval_img in exp.eval_list:
    filename = eval_img.split("/")[-1]
    label = filename.split("_")[0]
    os.rename(eval_img, f"{validation_dir}/{label}/{filename}")

parameters = exp.get_data("parameters")

BATCH_SIZE = parameters["batch_size"]
IMG_SIZE = (parameters["image_size"], parameters["image_size"])

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)
validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)

class_names = train_dataset.class_names
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)

class PicselliaLogger(Callback):

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            exp.log(log_name, [float(log_value)], 'line')

callback_list = [PicselliaLogger()]

fine_tune = parameters["fine_tune"]
if not fine_tune:
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights=os.path.join(exp.base_dir, "model.h5"))

    base_model.trainable = False
    base_model.summary()

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=(parameters["image_size"], parameters["image_size"], 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = parameters["learning_rate"]
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])

    model.summary()

    initial_epochs = parameters["initial_epochs"]

    loss0, accuracy0 = model.evaluate(validation_dataset)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))
    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        callbacks=callback_list,
                        validation_data=validation_dataset)

    base_model.trainable = True
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
                metrics=['accuracy'])
    model.summary()


    fine_tune_epochs = parameters["fine_tune_epochs"]
    total_epochs =  initial_epochs + fine_tune_epochs

    history_fine = model.fit(train_dataset,
                            epochs=total_epochs,
                            initial_epoch=history.epoch[-1],
                            callbacks=callback_list,
                            validation_data=validation_dataset)

else:
    model = models.load_model(os.path.join(exp.base_dir, "model.h5"))
    history_fine = model.fit(train_dataset,
                            epochs=parameters["fine_tune_epochs"],
                            initial_epoch=0,
                            callbacks=callback_list,
                            validation_data=validation_dataset)

models_dir = os.path.join(exp.base_dir, 'models')
tf.saved_model.save(model, os.path.join(models_dir, 'saved_model'))
model.save(os.path.join(models_dir, 'keras_model/model.h5'))
exp.store("keras_model", os.path.join(models_dir, 'keras_model/model.h5'))
exp.store("model-latest", os.path.join(models_dir, 'saved_model'), zip=True)
parameters["fine_tune"] = True
exp.log("parameters", parameters, "table", replace=True)
exp.update(status="success")

y_true = []
y_pred = []
for btc in range(len(validation_dataset)):
    image_batch, label_batch = validation_dataset.as_numpy_iterator().next()
    predictions = model.predict_on_batch(image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)
    y_true.extend(label_batch)
    y_pred.extend(predictions.numpy())


print('Confusion Matrix')
print('----------------')
print(confusion_matrix(y_true, y_pred))
print()
print('Classification Report')
print('---------------------')
print(classification_report(y_true, y_pred, target_names=exp.categories))
cm = {
    'categories': exp.categories,
    'values': confusion_matrix(y_true, y_pred).tolist()
}
exp.log('confusion-matrix', cm, 'heatmap')