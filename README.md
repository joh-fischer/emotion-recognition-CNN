# Facial Emotion Recognition with CNNs

### Pre-trained Models

You can choose between two pre-trained models: First, the model trained on [RAF-DB](http://whdeng.cn/RAF/model1.html), and second, the model trained on [FER+](https://github.com/microsoft/FERPlus). The models' weights are stored in the `models` folder. We obtained the following results:

|  | RAF-DB model | FER+ model  |
| :------------- | :------------- | :------------- |
| Validation Accuracy | 82.99%   | 84.21% |
| Test Accuracy | 82.72% | 83.77% |

Below you can find the code which describes how to load and use each of the models.

```python
from model import get_base_model
from utils import preprocess_fer, get_labels_fer

import numpy as np
import tensorflow as tf
import cv2
```

**1. Build base model**

The base model is equal for both pre-trained models.

```python
IMG_SHAPE = (100, 100, 3)

model = get_base_model(IMG_SHAPE)
model.add(tf.keras.layers.Dense(7, activation='softmax', name="softmax"))
```

**2. Load weights**

Here you have to choose which model you want to take.
* RAF-DB model: `model_name = RAF_0124-1008_weights.h5`
* FER+ model: `model_name = FERplus_0124-1040_weights.h5`

```python
model_name = 'FERplus_0124-1040_weights.h5'   # FER+ example
model.load_weights('./models/' + model_name)
```

**3. Apply model**

Load any image of a cropped face, ensure that the shape is (100, 100, 3), preprocess it with the model dependent preprocessing function and feed it into the model. The output is a probability distribution indicating which emotion is most likely.

Example for the FER+ model:
```python
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
```


### Evaluation

The notebook `Model Evaluation.ipynb` helps you to evaluate the pre-trained models and visualize the classification results.

### Training

If you want to train the models from scratch, you first need to populate the `data` folder with the respective dataset ([FER+](https://github.com/microsoft/FERPlus) or [RAF-DB](http://whdeng.cn/RAF/model1.html)). The two jupyter notebooks `Model FER+.ipynb` and `Model RAF.ipynb` will then guide you through the training process.
