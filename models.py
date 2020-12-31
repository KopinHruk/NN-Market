from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf



def create_encoder(input_dim, output_dim, noise=0.05):
    i = Input(input_dim)
    encoded = BatchNormalization()(i)
    encoded = GaussianNoise(noise)(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.2)(encoded)

    encoded = Dense(64, activation='relu')(encoded)

    decoded = BatchNormalization()(encoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(input_dim, name='decoded')(decoded)

    x = Dense(32)(decoded)
    x = BatchNormalization()(x)
    x = Lambda(tf.keras.activations.swish)(x)
    x = Dropout(0.2)(x)
    x = Dense(16)(x)
    x = BatchNormalization()(x)
    x = Lambda(tf.keras.activations.swish)(x)
    x = Dropout(0.2)(x)
    x = Dense(output_dim, activation='sigmoid', name='label_output')(x)

    encoder = Model(inputs=i, outputs=decoded)
    autoencoder = Model(inputs=i, outputs=[decoded, x])

    autoencoder.compile(optimizer=Adam(0.01), loss={'decoded': 'mse', 'label_output': 'binary_crossentropy'})
    return autoencoder, encoder


def create_model(input_dim, output_dim, encoder):
    inputs = Input(input_dim)

    x = encoder(inputs)
    x = Concatenate()([x, inputs])  # use both raw and de-noised features
    x = BatchNormalization()(x)


    x = Dense(300)(x)
    x = BatchNormalization()(x)
    x = Lambda(tf.keras.activations.swish)(x)


    x = Dense(300)(x)
    x = BatchNormalization()(x)
    x = Lambda(tf.keras.activations.swish)(x)


    x = Dense(output_dim, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(0.01),
                  loss=BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.Precision(name='Precision'), tf.keras.metrics.Recall(name='Recall'), tf.keras.metrics.BinaryAccuracy(name='acc')])
    return model