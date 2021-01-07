import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import os

# load dataset
train = pd.read_csv('data/train/train.csv')
submission = pd.read_csv('data/sample_submission.csv')

train['GHI'] = train['DHI'] + train['DNI']

WINDOW_SIZE = 48

#%% phase 1: weather prediction
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index, WINDOW_SIZE):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])
    data = np.array(data)
    labels = np.array(labels)
    if FEATURES == 1:
        # univariate
        data = data.reshape(-1, history_size, 1)
    return data, labels

def normalize(df):
    df_n = (df - train_min) / (train_max - train_min)
    return df_n

def denormalize(arr, feature):
    arr = arr * (train_max[feature] - train_min[feature]) + train_min[feature]
    return arr

x_col =['GHI']
y_col = ['GHI']

# normalize
train_n = train.copy()
train_max = train.max(axis=0)
train_min = train.min(axis=0)
train_n = normalize(train)

dataset = train_n.loc[:,x_col].values
label = np.ravel(train.loc[:,y_col].values)

FEATURES = len(x_col)
past_history = WINDOW_SIZE
future_target = WINDOW_SIZE

### transform train
train_data, train_label = multivariate_data(dataset, label, 0,
                                                   None, past_history,
                                                   future_target, 1,
                                                   single_step=False)
train_data = train_data.transpose((0, 2, 1))
train_data = train_data.reshape(-1, past_history * FEATURES)
train_label = (train_label.mean(axis=1) < 100).astype(int)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import initializers
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

# GPU setting
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size = 0.1, random_state = 0)

model = Sequential([
    tf.keras.layers.Dense(past_history,
                         input_shape=X_train.shape[-1:],
                         kernel_initializer=initializers.he_normal(), activation='relu'),
    tf.keras.layers.Dense(1,
                          kernel_initializer=initializers.he_normal(), activation='sigmoid'),
])
es = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=15,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

def scheduler(epoch, lr):
  if epoch < 30:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

lr = LearningRateScheduler(scheduler)

optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_crossentropy'])
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_val, y_val), batch_size=128,
                    callbacks=[lr, es], verbose=2)
tf.keras.backend.clear_session()

y_pred = model.predict(X_val)
# denormalize
y_pred = denormalize(y_pred, y_col[0])
y_val = denormalize(y_val, y_col[0])

# binary classification error
y_pred_day1 = (y_pred[:,:48].mean(axis=1) < 50).astype(int)
y_pred_day2 = (y_pred[:,48:96].mean(axis=1) < 50).astype(int)
y_val_day1 = (y_val[:,:48].mean(axis=1) < 50).astype(int)
y_val_day2 = (y_val[:,48:96].mean(axis=1) < 50).astype(int)

(y_pred_day1 == y_val_day1).mean()
(y_pred_day2 == y_val_day2).mean()
