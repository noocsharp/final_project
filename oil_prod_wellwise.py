"""
Unlike oil_prod.py, this one trains one complete wells, rather than random points from random wells
It is kind of suprising how well (pun intended) it performs on test data, despite not having data from every well.
Also considering we are only using 50 epochs and a relatively fast learning rate, this data must have a lot of structure, meaning ML isn't strictly necessary, it just makes the job substantially easier.

Assuming that it isn't overfitting, the next thing to do is figure out whether we can get as good performance just from x,y values
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
    X_train = pd.concat(X_train)
    X_test = pd.concat(X_test)

# removes columns that might be redundant/problematic
    X_train = X_train.drop(columns=['water saturation', 'proppant weight (lbs)', 'pump rate (cubic feet/min)']).reset_index()
    X_test = X_test.drop(columns=['water saturation', 'proppant weight (lbs)', 'pump rate (cubic feet/min)']).reset_index()

# separates our dependent variable out
    y_train = X_train.pop('oil saturation')
    y_test = X_test.pop('oil saturation')

# sets up our the neural network
    model = K.models.Sequential([K.layers.Dense(first, input_shape=[7,])])

    for i in args:
        model.add(K.layers.Dense(i))

    model.add(K.layers.Dense(1))

    #model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    model.compile(optimizer=opt, loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=100, patience=10, 
            restore_best_weights=True)

# fits model to data
    history = model.fit(X_train, y_train, epochs=epochs_, validation_data=(X_test, y_test), callbacks=[early_stopping])

    return model.to_json()

optimizer = K.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=.999, amsgrad=False)

models = []
for i in range(10):
    info = run_model(optimizer, .5, 50, 10, 10, 10)
    models.append(info)

with open('models_wellwise.txt', 'w') as f:
    f.write(json.dumps(models))
