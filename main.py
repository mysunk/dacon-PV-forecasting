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

x_col =[ 'DHI', 'DNI', 'WS', 'RH', 'T', 'D_sum', 'TARGET']
y_col = ['TARGET']

dataset = train.loc[:,x_col].values
label = train.loc[:,y_col].values

FEATURES = len(x_col)
past_history = 48*7
future_target = 48*2
STEP = 1

### transform train
X_train, y_train = multivariate_data(dataset, label, 0,
                                                   None, past_history,
                                                   future_target, STEP,
                                                   single_step=False)
TRAIN_SPLIT = int(X_train.shape[0] * 0.8)
X_val, y_val = X_train[TRAIN_SPLIT:], y_train[TRAIN_SPLIT:]
X_train, y_train = X_train[:TRAIN_SPLIT], y_train[:TRAIN_SPLIT]

### transform test
test = []
for i in range(81):
    data = []
    tmp = pd.read_csv(f'data/test/{i}.csv')
    tmp['D_sum'] = tmp['DHI'] + tmp['DNI']
    tmp = tmp.loc[:, x_col].values
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

## save the result
submission.to_csv('submit/submit_11.csv')
# save corresponding val result
val_preds_df.to_csv('val/val_11.csv')
