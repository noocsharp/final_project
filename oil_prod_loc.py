"""
This one tries to predict oil production purely on x, y location.
Probably won't be as successful
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.callbacks import EarlyStopping
import random

def run_model(opt, split, epochs_, first, *args):
# loads all well log data into a list
    X_train = []
    X_test = []
    for i in os.listdir("data"):
        if i != 'well production.csv':
            if random.randint(0, 1):
                X_train.append(pd.read_csv("data/" + i))
            else: X_test.append(pd.read_csv("data/" + i))

# combines all well log data into a dataframe
    X_train = pd.concat(X_train).reset_index()
    X_test = pd.concat(X_test).reset_index()

# removes columns that might be redundant/problematic
    X_train = X_train[['easting', 'northing', 'oil saturation']]
    X_test = X_test[['easting', 'northing', 'oil saturation']]

    print(X_train.describe())
    print(X_test.describe())

# separates our dependent variable out
    y_train = X_train.pop('oil saturation')
    y_test = X_test.pop('oil saturation')

# sets up our the neural network
    model = K.models.Sequential([K.layers.Dense(first, input_shape=[2,])])

    for i in args:
        model.add(K.layers.Dense(i))

    #model.add(K.layers.Dense(1))

    #model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    model.compile(optimizer=opt, loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=.001, patience=10, 
            restore_best_weights=True)

# fits model to data
    history = model.fit(X_train, y_train, epochs=epochs_, validation_data=(X_test, y_test), callbacks=[early_stopping])

    print(len(X_train), len(X_test))
    return model.to_json()

optimizer = K.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=.999, amsgrad=False)

models = []
for i in range(10):
    info = run_model(optimizer, .5, 100, 1)
    #info = run_model(optimizer, .5, 100, 10, 10, 10)
    models.append(info)

with open('models_wellwise.txt', 'w') as f:
    f.write(json.dumps(models))
