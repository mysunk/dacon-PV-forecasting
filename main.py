import pandas as pd
import numpy as np

train = pd.read_csv('data/train/train.csv')
submission = pd.read_csv('data/sample_submission.csv')
submission.set_index('id',inplace=True)

train['D_sum'] = train['DHI'] + train['DNI']

# save val result format
save_num = 20

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
        data.append(np.ravel(dataset[indices].T))
        if single_step:
            labels.append(target[i+target_size])
        else:
          labels.append(target[i:i+target_size])
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

x_col =['DHI','DNI','D_sum','WS','RH','T','TARGET']
y_col = ['TARGET']

dataset = train.loc[:,x_col].values
label = np.ravel(train.loc[:,y_col].values)

FEATURES = len(x_col)
past_history = 48 * 3
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

#%% rf model
def pinball_loss(q,y_true, y_pred):
    idx1 = y_true >= y_pred
    idx2 = y_true < y_pred

    y_true_1, y_pred_1 = np.ravel(y_true[idx1]),np.ravel(y_pred[idx1])
    y_true_2, y_pred_2 = np.ravel(y_true[idx2]),np.ravel(y_pred[idx2])

    loss_1 = (y_true_1 - y_pred_1)*q
    loss_2 = (y_pred_2 - y_true_2) * (1 - q)

    loss = np.concatenate([loss_1, loss_2])
    return np.mean(loss)

from sklearn import ensemble
N_ESTIMATORS = 1000
rf = ensemble.RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                    random_state=0,
                                    max_depth = 5,
                                    max_features=1,
                                    criterion='mae',
                                    verbose=True,
                                    n_jobs=-1)  # Use maximum number of cores.

## LOOCV
sample_num = train_data.shape[0]
loss_list = np.zeros([sample_num, 1])
val_pred = [np.zeros(train_label.shape) for _ in range(9)]
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)

from tqdm import tqdm
for train_index, test_index in tqdm(kf.split(train_data)):  # cross validation
    X_train, X_test = train_data[train_index], train_data[test_index]
    y_train, y_test = train_label[train_index], train_label[test_index]

    rf.fit(X_train, y_train)

    rf_preds = []
    for estimator in rf.estimators_:
        rf_preds.append(estimator.predict(X_test))
    rf_preds = np.array(rf_preds)

    for i, q in enumerate(np.arange(0.1, 1, 0.1)):
        val_pred[i][test_index] = np.percentile(rf_preds, q * 100, axis=0)

# val result dictionary and loss
val_preds_df = pd.DataFrame(columns=list(submission.columns) + ['true'],
                           data=np.zeros((train_label.shape[0] * future_target,10)))
val_preds_df['true'] = np.ravel(train_label)
losses = []
for i, q in enumerate(np.arange(0.1, 1, 0.1)):
    val_preds_df.iloc[:, i] = np.ravel(val_pred[i])
    losses.append(pinball_loss(q, np.ravel(train_label), np.ravel(val_pred[i])))
loss = np.mean(losses, axis=0)
print(loss)

#%% test == 전체 데이터셋 사용
rf.fit(train_data, train_label)

rf_preds = []
for estimator in rf.estimators_:
    rf_preds.append(estimator.predict(test))
rf_preds = np.array(rf_preds)

for i, q in enumerate(np.arange(0.1, 1, 0.1)):
    y_pred = np.percentile(rf_preds, q * 100, axis=0)
    submission.iloc[:, i] = np.ravel(y_pred)

#%% save prediction and validation result
submission.to_csv(f'submit/submit_{save_num}.csv')
val_preds_df.to_csv(f'val/val_{save_num}.csv')

f = open(f"val/val_{save_num}.txt","w+")
msg = f'history length:: {past_history} \n'
f.write(msg)
msg = f'used feature:: {x_col} \n'
f.write(msg)
msg = f'pinball cv loss:: {loss} \n'
f.write(msg)
f.close()
