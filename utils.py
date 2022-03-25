import numpy as np
import cv2
import os
import tensorflow as tf
import time

def get_mean_RAF(data_dir='./data/RAF/images'):
    """
	Get mean per channel of RAF training data for later image normalization.
	
	Args:
		data_dir (str): Path where the images are stored.
	"""
    t0 = time.time()

    X_train = []

    for filename in os.listdir(data_dir):
        if not filename.startswith('train'):
            continue

        fullpath = os.path.join(data_dir, filename)
        X_train.append(
                tf.keras.preprocessing.image.img_to_array(
                    tf.keras.utils.load_img(
                        fullpath, grayscale=False, color_mode='rgb',
                        #target_size=IMG_SIZE, interpolation='nearest'
        )))

    X_train = np.array(X_train, dtype='double')			# caution: dtype must be more precise than float32!!!
    print(X_train.shape)

    print("Gathered %s images in %.5f seconds" % (X_train.shape[0], time.time() - t0))

    over_axis_mean = np.mean(X_train, axis=(0, 1, 2))
    print("Mean per channel: ", over_axis_mean)
    over_axis_std = np.std(X_train, axis=(0, 1, 2))
    print("Std per channel: ", over_axis_std)

    print('Gathered images and calculated statistics in %.4f seconds' % (time.time() - t0))


def get_mean_FERplus(data_dir='./data/ferplus2013/images/FER2013Train'):
    """
	Get mean per channel of FER+ training data for later image normalization.
	
	Args:
		data_dir (str): Path where the training images are stored.
	"""
    t0 = time.time()

    X_train = []
    for filename in os.listdir(data_dir):
        fullpath = os.path.join(data_dir, filename)
        X_train.append(
                tf.keras.preprocessing.image.img_to_array(
                    tf.keras.utils.load_img(
                        fullpath, grayscale=False, color_mode='rgb',
                        #target_size=IMG_SIZE, interpolation='nearest'
        )))

    X_train = np.array(X_train, dtype='double')			# caution: dtype must be more precise than float32!!!
    print(X_train.shape)

    print("Gathered %s images in %.5f seconds" % (X_train.shape[0], time.time() - t0))

    over_axis_mean = np.mean(X_train, axis=(0, 1, 2))
    print("Mean per channel: ", over_axis_mean)
    over_axis_std = np.std(X_train, axis=(0, 1, 2))
    print("Std per channel: ", over_axis_std)

    print('Gathered images and calculated statistics in %.4f seconds' % (time.time() - t0))


if __name__ == "__main__":
    print("--- FERplus:")
    get_mean_FERplus()

    print("\n--- RAF-DB:")
    get_mean_RAF()