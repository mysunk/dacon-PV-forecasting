import pandas as pd
import numpy as np

train = pd.read_csv('data/train/train.csv')
submission = pd.read_csv('data/sample_submission.csv')
submission.set_index('id',inplace=True)

train['D_sum'] = train['DHI'] + train['DNI']

#%% pre-processing
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index, 48):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i+target_size])
        else:
          labels.append(target[i:i+target_size])
    data = np.array(data)
    labels = np.array(labels)
    if FEATURES == 1:
        # univariate
        data = data.reshape(-1,history_size,1)
    return data, labels

used_features =[ 'DHI', 'DNI', 'WS', 'RH', 'T', 'D_sum', 'TARGET']
# used_features =['D_sum', 'TARGET']
dataset = train.loc[:,used_features].values
FEATURES = len(used_features)
past_history = 48*7
future_target = 48*2
STEP = 1

### train
X_train, y_train = multivariate_data(dataset, dataset[:, -1], 0,
                                                   None, past_history,
                                                   future_target, STEP,
                                                   single_step=False)
TRAIN_SPLIT = int(X_train.shape[0] * 0.8)
X_val, y_val = X_train[TRAIN_SPLIT:], y_train[TRAIN_SPLIT:]
X_train, y_train = X_train[:TRAIN_SPLIT], y_train[:TRAIN_SPLIT]

test = []
for i in range(81):
    data = []
    tmp = pd.read_csv(f'data/test/{i}.csv')
    tmp['D_sum'] = tmp['DHI'] + tmp['DNI']
    tmp = tmp.loc[:, used_features].values
    tmp = tmp[-past_history:,:]
    data.append(tmp)
    data = np.array(data)
    if FEATURES == 1:
        data = data.reshape(-1, past_history, 1)
    test.append(data)
test = np.concatenate(test, axis=0)

X_train = X_train.transpose((0,2,1))
X_val = X_val.transpose((0,2,1))
test = test.transpose((0,2,1))

X_train = X_train.reshape(-1,past_history*FEATURES)
X_val = X_val.reshape(-1,past_history*FEATURES)
test = test.reshape(-1,past_history*FEATURES)

#%% rf model
X_train = X_train.transpose((0,2,1))
X_val = X_val.transpose((0,2,1))
test = test.transpose((0,2,1))

X_train = X_train.reshape(-1,past_history*FEATURES)
X_val = X_val.reshape(-1,past_history*FEATURES)
test = test.reshape(-1,past_history*FEATURES)

from sklearn import ensemble
N_ESTIMATORS = 1000
rf = ensemble.RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                    min_samples_leaf=1, random_state=3,
                                    verbose=True,
                                    n_jobs=-1)  # Use maximum number of cores.

rf.fit(X_train, y_train)

## val
rf_preds = []
for estimator in rf.estimators_:
    rf_preds.append(estimator.predict(X_val))
rf_preds = np.array(rf_preds)

val_preds_df = pd.DataFrame(columns=list(submission.columns) + ['true'],
                           data=np.zeros((y_val.shape[0] * future_target,10)))
val_preds_df['true'] = np.ravel(y_val)

for i, q in enumerate(np.arange(0.1, 1, 0.1)):
    val_pred = np.percentile(rf_preds, q * 100, axis=0)
    val_preds_df.iloc[:,i] = np.ravel(val_pred)

## test
rf_preds = []
for estimator in rf.estimators_:
    rf_preds.append(estimator.predict(test))
rf_preds = np.array(rf_preds)

for i, q in enumerate(np.arange(0.1, 1, 0.1)):
    y_pred = np.percentile(rf_preds, q * 100, axis=0)
    submission.iloc[:, i] = np.ravel(y_pred)

submission.to_csv('submit/submit_11.csv')
# save corresponding val result
val_preds_df.to_csv('val/val_11.csv')

#%% evaluation
import tensorflow.keras.backend as K
def pinball(tao, y_true, y_pred):
    pin = K.mean(K.maximum(y_true - y_pred, 0) * tao +
                 K.maximum(y_pred - y_true, 0) * (1 - tao))
    return pin

import matplotlib.pyplot as plt
val_preds_df = pd.read_csv('val/val_11.csv')
val_preds_df.set_index('Unnamed: 0',inplace=True)

val_results = []
val_preds_df[val_preds_df<0] = 0
for q in np.arange(0.1,1,0.1):
    val = pinball(q,val_preds_df['true'], val_preds_df['q_'+str(q)[:3]])
    val_results.append(val)
print(np.mean(val_results, axis=0))

plt.plot(val_preds_df.iloc[:48,9],'r',label='True')
for i in range(9):
    plt.plot(val_preds_df.iloc[:48,i],'x',label=val_preds_df.columns[i])
plt.legend()
plt.show()