# Trains a 'teacher' (cumbersome) model on MNIST

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

seed = 0
epochs = 200
model_path = "../tf_models/mnist_teacher_epochs{}".format(epochs)


class hinton(tf.keras.Model):
    def __init__(self, temperature=1.0):
        super(hinton, self).__init__()
        self.temperature = temperature
        self.net = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            # keras.layers.Dropout(0.2),
            keras.layers.Dense(1200, activation='relu'),
            keras.layers.BatchNormalization(),
            # keras.layers.Dropout(0.5),
            keras.layers.Dense(1200, activation='relu'),
            keras.layers.BatchNormalization(),
            # keras.layers.Dropout(0.5),
            keras.layers.Dense(10)
        ])

    def call(self, X):
        X = tf.convert_to_tensor(X, dtype='float32')
        eta = self.net(X) / self.temperature
        return tf.math.softmax(eta)

if __name__ == '__main__':

    mnist = keras.datasets.mnist

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.astype('float32') / 255.
    test_images = test_images.astype('float32') / 255.

    tf.random.set_seed(seed)
    model = hinton()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images,
              train_labels,
              epochs=epochs,
              validation_data=(test_images,test_labels)
              )

    model.save_weights(model_path)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
