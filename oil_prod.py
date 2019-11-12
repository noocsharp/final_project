import os
import numpy as np
import pandas as pd
import tensorflow as tf

# loads all well log data into a list
data = []
for i in os.listdir("data"):
    if i != 'well production.csv':
        data.append(pd.read_csv("data/" + i))

# combines all well log data into a dataframe
combined = pd.concat(data)

# removes columns that might be redundant/problematic
combined = combined.drop(columns=['water saturation', 'proppant weight (lbs)', 'pump rate (cubic feet/min)']).reset_index()

train = combined.sample(frac=0.5)
test = combined.drop(train.index)

# separates our dependent variable out
y_train = train.pop('oil saturation')
y_test = test.pop('oil saturation')

# sets up our the neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[7,]),
    tf.keras.layers.Dense(4),
    tf.keras.layers.Dense(1)
    ])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# fits model to data
model.fit(train, y_train, epochs=10)

y_pred = model.predict(test)

test_error = int(tf.keras.losses.MeanSquaredError()(y_test, y_pred))
print(test_error)
