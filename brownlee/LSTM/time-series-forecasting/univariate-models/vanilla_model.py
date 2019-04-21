import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# Split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of the pattern
        end_ix = i + n_steps
        if(end_ix > len(sequence) - 1):
            break
        
        # gather the input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# define input sequence
raw_seq = [10, 21, 32, 43, 54, 65, 76, 87, 98]
new_seq = list()
for i in range(1, 100):
    new_seq.append(i*20)

# choose a number of timestamps
n_steps = 3

# Split into samples
X, y = split_sequence(new_seq, n_steps)

# reshape from [samples, timestamps] into [samples, timestamp, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model
model = Sequential()
model.add(LSTM(50, activation="relu", input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer = "adam", loss = "mse")

#fit model
model.fit(X, y, batch_size=100, epochs=100, verbose=0)

# demostrate prediction
X_input = np.array([60, 80, 100]) #103
X_input = X_input.reshape((1, n_steps, n_features))
yhat = model.predict(X_input, verbose = 0)

print(yhat)
# 104.14432