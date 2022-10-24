from PIL import Image
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

def dataset(experiment, assets, count, split_type, new_size, n_classes):

    X = []
    for i in range(len(assets)):
        path = os.path.join(experiment.png_dir, split_type, assets[i].filename)
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
