import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.callbacks import EarlyStopping

def run_model(opt, split, epochs_, first, *args):
# loads all well log data into a list
    data = []
    for i in os.listdir("data"):
        if i != 'well production.csv':
            data.append(pd.read_csv("data/" + i))

# combines all well log data into a dataframe
    combined = pd.concat(data)

# removes columns that might be redundant/problematic
    X = combined.drop(columns=['water saturation', 'proppant weight (lbs)', 'pump rate (cubic feet/min)']).reset_index()

# separates our dependent variable out
    y = X.pop('oil saturation')

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
    history = model.fit(X, y, epochs=epochs_, validation_split=split, callbacks=[early_stopping])

    return model.to_json()

optimizer = K.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=.999, amsgrad=False)

models = []
for i in range(10):
    info = run_model(optimizer, .5, 50, 10, 10, 10)
    models.append(info)

with open('models.txt', 'w') as f:
    f.write(json.dumps(models))
