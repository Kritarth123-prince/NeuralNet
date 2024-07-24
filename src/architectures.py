import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

def create_cnn(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def create_rnn(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=10000, output_dim=32, input_length=input_shape))
    model.add(layers.SimpleRNN(32))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def create_gan(input_shape, noise_dim):
    # Generator
    generator = models.Sequential()
    generator.add(layers.Dense(256, activation='relu', input_dim=noise_dim))
    generator.add(layers.BatchNormalization(momentum=0.8))
    generator.add(layers.Reshape((8, 8, 4)))
    generator.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    generator.add(layers.ReLU())
    generator.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    generator.add(layers.ReLU())
    generator.add(layers.Conv2D(1, (7, 7), activation='tanh', padding='same'))

    # Discriminator
    discriminator = models.Sequential()
    discriminator.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Dropout(0.25))
    discriminator.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    discriminator.add(layers.LeakyReLU(alpha=0.2))
    discriminator.add(layers.Dropout(0.25))
    discriminator.add(layers.Flatten())
    discriminator.add(layers.Dense(1, activation='sigmoid'))

    # GAN
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False
    gan_input = layers.Input(shape=(noise_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = models.Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')

    return generator, discriminator, gan