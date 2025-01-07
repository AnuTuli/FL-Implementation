#Communication with Server for model initialization in DFL

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from tensorflow.keras.utils import to_categorical
import pandas as pd
import socket
import tqdm
import json
import os
from _thread import *
import threading
from mlsocket import MLSocket

SEPARATOR = "<SEPARATOR>"
BUFFER_SIZE = 16384

print_lock = threading.Lock()

# Load global data
global_data_path = 'cropser.csv'
global_data = pd.read_csv(global_data_path)
X_global = global_data.iloc[:, 0:7].astype(np.float32).values
y_global = global_data.iloc[:, 7]

lb = LabelEncoder()
y_global = lb.fit_transform(y_global).astype(np.int32)
y_global = to_categorical(y_global)

sc = StandardScaler()
X_global = sc.fit_transform(X_global)
X_global = X_global.reshape((X_global.shape[0], 1, X_global.shape[1]))  # Reshape for LSTM

X_train, X_test, y_train, y_test = train_test_split(X_global, y_global, test_size=0.2, random_state=41)

#model_params={'Sequential()', 'LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]))','Dense(y_train.shape[1], activation=softmax)','optimizer=adam, loss=categorical_crossentropy, metrics=[accuracy]'}
#model_params=['start',50,'softmax','adam','categorical_crossentropy','end']
# global model training using LSTM

SERVER_HOST = "10.0.4.102"
SERVER_PORT = 65432
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"

local_weights=[]
all_addr=[]

def init_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_global.shape[1], X_global.shape[2])))
    model.add(Dense(y_global.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    

t=0

s = MLSocket()
s.bind((SERVER_HOST, SERVER_PORT))
max_it=2
j=0

def send_model():
    model=init_model()
    model.fit(X_train, y_train, epochs=30, batch_size=32)
    all_addr=[]
    while True:
        s.listen(2)
        print(f"[*] Listening as {SERVER_HOST}:{SERVER_PORT}")
        print("Waiting for the client to connect... ")
        client_socket, addr=s.accept()
        all_addr.append(client_socket)
        print(f"[+] {addr} is connected.")
        client_socket.send(model)
        print("Model sent.")

send_model()
s.close()


