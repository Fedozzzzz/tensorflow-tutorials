from abc import ABC
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, losses

# load dataset and change to desired format
mnist = datasets.mnist

(train_images, _), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

test_labels = test_labels.astype(bool)

# plotting first 10 images compared
def plot_compared_images(images, recons, title):
    MAX_N = 10
    n = min(len(images), MAX_N)
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original + noise
        ax = plt.subplot(2, n, i + 1)
        plt.title("original " + title)
        plt.imshow(tf.squeeze(images[i]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        bx = plt.subplot(2, n, i + n + 1)
        plt.title("recons. " + title)
        plt.imshow(tf.squeeze(recons[i]))
        plt.gray()
        bx.get_xaxis().set_visible(False)
        bx.get_yaxis().set_visible(False)
    plt.show()


def get_threshold():
    errors = []
    # loop over all original images and their corresponding
    # reconstructions
    recons = autoencoder.call(train_images)
    for (image, recon) in zip(train_images, recons):
        # compute the mean squared error between the ground-truth image
        # and the reconstructed image, then add it to our list of errors
        mse = np.mean((image - recon) ** 2)
        errors.append(mse)

    # define threshold
    return np.quantile(errors, 0.99)


# MSE anomaly detecting
def detect_anomalies(images, recons):
    errors = []
    # loop over all original images and their corresponding
    # reconstructions
    for (image, recon) in zip(images, recons):
        # compute the mean squared error between the ground-truth image
        # and the reconstructed image, then add it to our list of errors
        mse = np.mean((image - recon) ** 2)
        errors.append(mse)

    # define threshold
    # threshold = np.quantile(errors, 0.99)
    threshold = get_threshold()
    idxs = np.where(np.array(errors) >= threshold)[0]
    print("[INFO] mse threshold: {}".format(threshold))
    print("[INFO] {} outliers found".format(len(idxs)))
    return idxs


class AnomalyDetector(Model, ABC):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
            # layers.Flatten(),
            # layers.Dense(512, activation="relu"),
            # layers.Dense(8, activation="relu"),
            # layers.Dense(128, activation="relu"),
            # layers.Dense(64, activation="relu"),
        ])

        self.decoder = tf.keras.Sequential([
            # layers.Dense(64, activation="relu"),
            # layers.Dense(128, activation="relu"),
            # layers.Dense(256, activation="relu"),
            # layers.Dense(512, activation="relu"),
            # layers.Dense(784, activation="sigmoid"),
            # layers.Reshape((28, 28, 1)),
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = AnomalyDetector()

# compile autoencoder
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])

# train autoencoder
history = autoencoder.fit(
    train_images,
    train_images,
    epochs=20,
    batch_size=784,
    validation_data=(test_images, test_images),
    shuffle=True, )

encoded_imgs = autoencoder.encoder(test_images).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

# plot testing images in 'handwritten numbers' dataset
plot_compared_images(test_images, decoded_imgs, 'numbers')

# load 'fashion' dataset
(_, _), (anomalous_test_data, anomalous_test_labels) = fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
anomalous_test_data = anomalous_test_data / 255.
anomalous_test_data = anomalous_test_data.reshape(-1, 28, 28, 1)

decoded_anomalous_test_imgs = autoencoder.call(anomalous_test_data).numpy()

# decoding 'fashion' images dataset
plot_compared_images(anomalous_test_data, decoded_anomalous_test_imgs, 'fashion')

# Anomaly detection
# Concatenate two datasets (500 images from 'fashion', 500 images from 'handwritten')
test_set_with_anomalies = np.concatenate((tf.squeeze(test_images[0:500]), tf.squeeze(anomalous_test_data[0:500])),
                                         axis=0)
np.random.shuffle(test_set_with_anomalies)
test_set_with_anomalies = test_set_with_anomalies.reshape(-1, 28, 28, 1)

decoded_anomalies_imgs = autoencoder.call(test_set_with_anomalies).numpy()

# get indexes of detected anomalies array
anomalies_idxs = detect_anomalies(test_set_with_anomalies, decoded_anomalies_imgs)

detected_anomalies_recons = []
detected_anomaly_original = []

# find anomalies in dataset
for i in range(len(anomalies_idxs)):
    index = anomalies_idxs[i]
    detected_anomalies_recons.append(decoded_anomalies_imgs[index])
    detected_anomaly_original.append(test_set_with_anomalies[index])

# plot detected anomalies
plot_compared_images(detected_anomaly_original, detected_anomalies_recons, 'anomalies')
