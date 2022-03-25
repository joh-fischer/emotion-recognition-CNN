# Explainable AI

**Contributors:**
* Hyeri An
* Anna Simon
* Michael Neumayr
* Johannes S. Fischer

### Loading Pre-trained Models

You can choose between two pre-trained models: First, the model trained on RAF-DB model, and second, the model trained on FER+. The models are stored in the relative path ```models``` folder. Below you can find the code which describes how to load each of the models. For each model the respective preprocessing function and emotion labels are given, as well as the name of the last convolutional layer used for visualization.

|  | RAF-DB model | FER+ model  |
| :------------- | :------------- | :------------- |
| Validation Accuracy | $82.99$%   | $84.21$% |
| Test Accuracy | $82.72$% | $83.77$% |


#### RAF-DB model

```py
from ModelGenerator import get_base_model2

# image shape
IMG_SHAPE = (100, 100, 3)

# model
model_name = 'RAF-impr-std_0124-1008_weights.h5'
model = get_base_model2(IMG_SHAPE)
model.add(tf.keras.layers.Dense(7, activation='softmax', name="softmax"))
model.load_weights('./models/' + model_name)

# last convolutional name
LAST_CONV_NAME = 'block3_conv3'

# preprocessing function for model
def preprocess(x):
    mean = [146.6770, 114.6274, 102.3102]
    std = [67.6282, 61.7651, 61.3665]
    # ensure image format
    x = np.array(x, dtype='float32')
    # normalize
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2]
    return x

emotion_labels = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
```

#### FER+ model

Model trained on the
```py
from ModelGenerator import get_base_model2

# image shape
IMG_SHAPE = (100, 100, 3)

# model
model_name = 'FERplus-impr-std_0124-1040_weights.h5'
model = get_base_model2(IMG_SHAPE)
model.add(tf.keras.layers.Dense(7, activation='softmax', name="softmax"))
model.load_weights('./models/' + model_name)

# last convolutional name
LAST_CONV_NAME = 'block3_conv3'

# preprocessing function for model
def preprocess(x):
    mean = [129.4432, 129.4432, 129.4432]
    std = [64.87448751, 64.87448751, 64.87448751]
    # ensure image format
    x = np.array(x, dtype='float32')
    # normalize
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2]
    return x

emotion_labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
```

### Evaluation

The notebook ```Model Evaluation.ipynb``` helps you to evaluate the pre-trained models and visualize the classification results.

### Training the Models

If you want to train the models by yourself, you need to fill the ```data``` folder with the respective dataset ([FER+](https://github.com/microsoft/FERPlus) or [RAF-DB](http://whdeng.cn/RAF/model1.html)). Then you can have a look at the two jupyter notebooks ```Model FER+.ipynb``` and ```Model RAF.ipynb```, which guide you through the training process.



----
If you have further questions or comments feel free to contact us. :)
