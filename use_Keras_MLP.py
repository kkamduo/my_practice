import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, input_shape=(3,))
])