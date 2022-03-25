from model import get_base_model
from utils import preprocess_fer, get_labels_fer

import numpy as np
import tensorflow as tf
import cv2


########
## 1. ##
########
IMG_SHAPE = (100, 100, 3)

model = get_base_model(IMG_SHAPE)
model.add(tf.keras.layers.Dense(7, activation='softmax', name="softmax"))

########
## 2. ##
########
model_name = 'FERplus_0124-1040_weights.h5'   # FER+ example
model.load_weights('./models/' + model_name)


########
## 3. ##
########
# load image
img = cv2.imread('./data/happy.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# preprocessing
x = cv2.resize(img, dsize=IMG_SHAPE[:-1])
x = np.expand_dims(x, axis=0)
x = preprocess_fer(x)

output = model.predict(x)

# get results
label = get_labels_fer(output)[0]
confidence = np.argmax(output[0])

print("Predicted class '{}' with confidence {:.2f}".format(label, confidence*100))