import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import os
# Seed value (can actually be different for each attribution step)
seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value) # tensorflow 2.x

def pinball(y_true, y_pred, q):
    pin = K.mean(K.maximum(y_true - y_pred, 0) * q +
                 K.maximum(y_pred - y_true, 0) * (1 - q))
    return pin

def load_obj(name):
    with open('tune_results/' + name + '.pkl', 'rb') as f:
        trials = pickle.load(f)
        trials = sorted(trials, key=lambda k: k['loss'])
        return trials

# load tuning results
# rf_param = load_obj('rf_3')[0]['params']
dnn_param = load_obj('dnn_3')[0]['params']
weight = [1,0]
lags = 3

# load dataset
train = pd.read_csv('data/train/train.csv')
submission = pd.read_csv('data/sample_submission.csv')
submission.set_index('id',inplace=True)

train['D_sum'] = train['DHI'] + train['DNI']

# save val result format
save_num = 21

#%% pre-processing
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index, 48):
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

x_col =['DHI','DNI','D_sum','WS','RH','T','TARGET']
y_col = ['TARGET']

dataset = train.loc[:,x_col].values
label = np.ravel(train.loc[:,y_col].values)

FEATURES = len(x_col)
past_history = 48 * lags
future_target = 48 * 2

### transform train
train_data, train_label = multivariate_data(dataset, label, 0,
                                                   None, past_history,
                                                   future_target, 1,
                                                   single_step=False)
### transform test
test = []
for i in range(81):
    data = []
    tmp = pd.read_csv(f'data/test/{i}.csv')
    tmp['D_sum'] = tmp['DHI'] + tmp['DNI']
    tmp = tmp.loc[:, x_col].values
    tmp = tmp[-past_history:,:]
    data.append(np.ravel(tmp.T))
    data = np.array(data)
    test.append(data)
test = np.concatenate(test, axis=0)

import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size = 0.2, random_state = 42)

# pre-precessing
X_train = X_train.transpose((0, 2, 1))
X_val = X_val.transpose((0, 2, 1))
X_train = X_train.reshape(-1, past_history * FEATURES)
X_val = X_val.reshape(-1, past_history * FEATURES)

#%% rf model
rf = RandomForestRegressor(**rf_param)
rf.fit(X_train, y_train)

rf_preds = []
for estimator in rf.estimators_:
    rf_preds.append(estimator.predict(X_val))
rf_preds = np.array(rf_preds)

val_results = pd.DataFrame(columns=['rf','dnn'], data = np.zeros((9,2)))
rf_preds_df = pd.DataFrame(columns=list(submission.columns) + ['true'],
                           data=np.zeros((y_val.shape[0] * future_target,9)))
for i, q in enumerate(np.arange(0.1, 1, 0.1)):
    val_pred = np.percentile(rf_preds, q * 100, axis=0)
    val = pinball(y_val, val_pred, q)
    val_results.loc[i,'rf'] = val
    rf_preds_df.iloc[:, i] = np.ravel(val_pred)

rf_preds_df['true'] = np.ravel(y_val)
print('rf')
print(np.mean(val_results['rf']))

#%% dnn model
def custom_loss(q):
    def pinball(y_true, y_pred):
        pin = K.mean(K.maximum(y_true - y_pred, 0) * q +
                     K.maximum(y_pred - y_true, 0) * (1 - q))
        return pin
    return pinball

def dnn_val(X_train, y_train, X_val, y_val, params, q):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(params['h1'], activation='relu', input_shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dense(params['h2'], activation='relu'))
    model.add(tf.keras.layers.Dense(params['h1'], activation='relu'))
    model.add(tf.keras.layers.Dense(future_target))

    optimizer = tf.keras.optimizers.Adam(params['lr'])
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=0,
        mode='auto',
        restore_best_weights=True
    )
    model.compile(optimizer=optimizer, loss=custom_loss(q=q), metrics=[custom_loss(q=q)])
    history = model.fit(X_train, y_train, epochs=params['EPOCH'], verbose=1, batch_size=params['BATCH_SIZE'],
                        validation_data=(X_val, y_val), callbacks=[es])
    y_pred = model.predict(X_val)
    # validation
    return np.min(history.history['val_loss']), y_pred, model

dnn_preds_df = pd.DataFrame(columns=list(submission.columns) + ['true'],
                           data=np.zeros((y_val.shape[0] * future_target,10)))

for i, q in enumerate(np.arange(0.1, 1, 0.1)):
    g = tf.Graph()
    with g.as_default():
        val, val_pred, _ = dnn_val(X_train, y_train, X_val, y_val, dnn_param, q)
        val_results.loc[i,'dnn'] = val
        dnn_preds_df.iloc[:, i] = np.ravel(val_pred)
    tf.keras.backend.clear_session()

dnn_preds_df['true'] = np.ravel(y_val)

print('dnn')
print(np.mean(val_results['dnn']))

#%% ensemble 성능 검증
losses = []
for i, q in enumerate(np.arange(0.1, 1, 0.1)):
    pred = rf_preds_df.iloc[:,i] * weight[0] + dnn_preds_df.iloc[:,i] * weight[1]
    loss = pinball(rf_preds_df['true'].values, pred, q)
    losses.append(loss)
print(np.mean(losses))

#%% test == 전체 데이터셋 사용
# rf
rf = RandomForestRegressor(**rf_param)
rf.fit(train_data, train_label)

rf_preds = []
for estimator in rf.estimators_:
    rf_preds.append(estimator.predict(test))
rf_preds = np.array(rf_preds)

rf_preds_test = []
for i, q in enumerate(np.arange(0.1, 1, 0.1)):
    y_pred = np.percentile(rf_preds, q * 100, axis=0)
    rf_preds_test.append(np.ravel(y_pred))

# dnn
dnn_preds_test = []
for i, q in enumerate(np.arange(0.1, 1, 0.1)):
    g = tf.Graph()
    with g.as_default():
        _, _, model = dnn_val(X_train, y_train, X_val, y_val, dnn_param, q)
        y_pred = model.predict(test)
        dnn_preds_test.append(np.ravel(y_pred))
    tf.keras.backend.clear_session()

# ensemble
for i in range(10):
    submission.loc[:,i] = rf_preds_test[i] * weight[0] + dnn_preds_test[i] * weight[1]

#%% save prediction and validation result
submission.to_csv(f'submit/submit_{save_num}.csv')
rf_preds_df.to_csv(f'val/rf_val_{save_num}.csv')
dnn_preds_df.to_csv(f'val/dnn_val_{save_num}.csv')

f = open(f"val/val_{save_num}.txt","w+")
msg = f'history length:: {past_history} \n'
f.write(msg)
msg = f'used feature:: {x_col} \n'
f.write(msg)
msg = f'pinball loss rf:: '+str(np.mean(val_results['rf']))+'\n'
f.write(msg)
msg = f'pinball loss dnn:: '+str(np.mean(val_results['dnn']))+'\n'
f.write(msg)
msg = f'pinball loss ens:: {np.mean(losses)} \n'
f.write(msg)
msg = f'used weight:: {weight} \n'
f.write(msg)
f.close()