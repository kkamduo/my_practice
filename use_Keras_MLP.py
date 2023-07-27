import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1,1],[1,0],[0,1],[0,0]])
y = np.array([[0],[1],[1],[0]])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='sigmoid', input_shape=(2,)), # First Hidden Layer
    tf.keras.layers.Dense(units=1, activation='sigmoid'), # Second Hidden Layer
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss='mse'
)
model.summary

history = model.fit(x,y, epochs=100, batch_size=1)

plt.plot(history.history['loss'])
plt.show()