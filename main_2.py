import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

WINDOW_SIZE = 48

#%% 1. load dataset
# train dataset
train = pd.read_csv('data/train/train.csv')
train['D_sum'] = train['DHI'] + train['DNI']
train_list = []
train_label_list = []
days = train['Day'].iloc[-1]
for i in range(0, days-7):
    tmp = train.iloc[i*WINDOW_SIZE:(i+7)*WINDOW_SIZE,:]
    tmp2 = train.iloc[(i+7)*WINDOW_SIZE:(i+9)*WINDOW_SIZE,:]
    train_list.append(tmp)
    train_label_list.append(tmp2)

# test dataset
test_list = []
for i in range(81):
    data = []
    tmp = pd.read_csv(f'data/test/{i}.csv')
    tmp['D_sum'] = tmp['DHI'] + tmp['DNI']
    test_list.append(tmp)

# submission file
submission = pd.read_csv('data/sample_submission.csv')
submission.set_index('id',inplace=True)

# normalize
train_max = train.max(axis=0)
train_min = train.min(axis=0)

#%% 2. clustering validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, davies_bouldin_score, silhouette_samples
from sklearn.cluster import KMeans

def normalize(df):
    df_n = (df - train_min) / (train_max - train_min)
    return df_n

def denormalize(arr):
    arr = arr * (train_max['TARGET'] - train_min['TARGET']) + train_min['TARGET']
    return arr

def transform(data_list, features, feature_weight, lags = 7, norm = True):
    arr = np.zeros((len(data_list),lags * 3 * len(features)))
    for i, data in enumerate(data_list):
        if norm:
            data = normalize(data)
        data = data.iloc[-lags * WINDOW_SIZE:, :]
        data = data.loc[:, features].copy()
        data = data * feature_weight
        tmp_arr = []
        for feature in features:
            tmp = data[feature].values.reshape(lags,-1)
            mean_ = tmp.mean(axis=1)
            max_ = tmp.max(axis=1)
            min_ = tmp.min(axis=1)
            tmp_arr.append(np.concatenate([mean_, max_, min_]))
        tmp_arr = np.concatenate(tmp_arr)
        arr[i] = tmp_arr
    return arr

def pinball_loss(y_true, y_pred, q):
    idx1 = y_true >= y_pred
    idx2 = y_true < y_pred

    y_true_1, y_pred_1 = np.ravel(y_true[idx1]),np.ravel(y_pred[idx1])
    y_true_2, y_pred_2 = np.ravel(y_true[idx2]),np.ravel(y_pred[idx2])

    loss_1 = (y_true_1 - y_pred_1)*q
    loss_2 = (y_pred_2 - y_true_2) * (1 - q)

    loss = np.concatenate([loss_1, loss_2])
    return np.mean(loss)

# train-val split
tr_list, val_list, tr_label_list, val_label_list = \
    train_test_split(train_list, train_label_list,test_size = 0.2, random_state = 4)

# transform with weather features
# WEATHER_FEATURES = ['D_sum','WS','RH','T']
# feature_weight = [0.4, 0.2, 0.2, 0.2]
WEATHER_FEATURES = ['TARGET']
feature_weight = [1]
weather_noem = True
lags = 2
train_w = transform(tr_list, features=WEATHER_FEATURES,feature_weight = feature_weight, lags=lags, norm = weather_noem)
val_w = transform(val_list, features=WEATHER_FEATURES,feature_weight = feature_weight, lags=lags, norm = weather_noem)
test_w = transform(test_list, features=WEATHER_FEATURES,feature_weight = feature_weight, lags=lags, norm = weather_noem)

# clustering
db_index, s_index = [], []
for num_clusters in range(2, 10):
    kmeans = KMeans(n_clusters = num_clusters, random_state=0)
    kmeans.fit(train_w)
    train_cluster_label = kmeans.labels_
    val_cluster_label = kmeans.predict(val_w)
    test_cluster_label = kmeans.predict(test_w)
    print('DB index: {:.2f}'.format(davies_bouldin_score(train_w, kmeans.labels_)))
    db_index.append(davies_bouldin_score(train_w, kmeans.labels_))
    print('Silhouette index: {:.2f}'.format(silhouette_samples(train_w, kmeans.labels_).mean()))
    s_index.append(silhouette_samples(train_w, kmeans.labels_).mean())
    # print(np.unique(train_cluster_label, return_counts=True))
# del train_w, val_w
plt.plot(db_index)
plt.plot(s_index)
plt.show()

#%% 2-2. clustering
num_clusters = 2
kmeans = KMeans(n_clusters = num_clusters, random_state=0)
kmeans.fit(train_w)
train_cluster_label = kmeans.labels_
val_cluster_label = kmeans.predict(val_w)
test_cluster_label = kmeans.predict(test_w)
print('DB index: {:.2f}'.format(davies_bouldin_score(train_w, kmeans.labels_)))
print('Silhouette index: {:.2f}'.format(silhouette_samples(train_w, kmeans.labels_).mean()))
print(np.unique(train_cluster_label, return_counts=True))

#%% 3. clustering based forecasting
def transform(data_list, features, lags = 7, norm = False):
    arr = np.zeros((len(data_list),lags * WINDOW_SIZE * len(features)))
    for i, data in enumerate(data_list):
        if norm:
            data = normalize(data)
        data = data.iloc[-lags * WINDOW_SIZE:,:]
        data = data.loc[:,features]
        tmp_arr = []
        for feature in features:
            tmp = data[feature].values
            tmp_arr.append(tmp)
        tmp_arr = np.concatenate(tmp_arr)
        arr[i] = tmp_arr
    return arr

# transform with weather and target
is_norm = False
train_arr = transform(tr_list, features=['TARGET'], lags=2, norm=is_norm)
val_arr = transform(val_list, features=['TARGET'], lags=2, norm=is_norm)
train_label_arr = transform(tr_label_list, features=['TARGET'], lags=2, norm=is_norm)
val_label_arr = transform(val_label_list, features=['TARGET'], lags=2, norm=is_norm)

# just forecasting
from sklearn.ensemble import RandomForestRegressor
rf_param = {'criterion': 'mae', 'n_jobs':-1, 'max_features': 1, 'max_depth': 5}
rf_global = RandomForestRegressor(**rf_param)
rf_global.fit(train_arr, train_label_arr)

# clustering based forecasting
cluster_mae, globel_mae = [], []
for i in range(num_clusters):
    print(f'For cluster {i}')
    # filter data
    train_arr_filtered = train_arr[train_cluster_label == i,:]
    train_label_arr_filtered = train_label_arr[train_cluster_label == i, :]
    val_arr_filtered = val_arr[val_cluster_label == i, :]
    val_label_arr_filtered = val_label_arr[val_cluster_label == i, :]

    # clustering based forecasting
    rf = RandomForestRegressor(**rf_param)
    rf.fit(train_arr_filtered, train_label_arr_filtered)

    rf_preds = []
    for estimator in rf.estimators_:
        rf_preds.append(estimator.predict(val_arr_filtered))
    rf_preds = np.array(rf_preds)

    rf_preds_1 = []
    for i, q in enumerate(np.arange(0.1, 1, 0.1)):
        y_pred = np.percentile(rf_preds, q * 100, axis=0)
        rf_preds_1.append(y_pred)

    # compare with global predict
    rf_preds = []
    for estimator in rf_global.estimators_:
        rf_preds.append(estimator.predict(val_arr_filtered))
    rf_preds = np.array(rf_preds)

    rf_preds_2 = []
    for i, q in enumerate(np.arange(0.1, 1, 0.1)):
        y_pred = np.percentile(rf_preds, q * 100, axis=0)
        rf_preds_2.append(y_pred)

    # mae
    if is_norm:
        val_label_arr_filtered = denormalize(val_label_arr_filtered)
        # FIXME
        val_pred_1 = denormalize(val_pred_1)
        val_pred_2 = denormalize(val_pred_2)

    losses_1, losses_2 = [], []
    for i, q in enumerate(np.arange(0.1, 1, 0.1)):
        loss1 = pinball_loss(val_label_arr_filtered, rf_preds_1[i], q)
        loss2 = pinball_loss(val_label_arr_filtered, rf_preds_2[i], q)
        losses_1.append(loss1)
        losses_2.append(loss2)

    mae_1, mae_2 = np.mean(losses_1), np.mean(losses_1)
    print('clustering based:{:.2f}'.format(np.mean(losses_1)))
    print('global:{:.2f}'.format(np.mean(losses_2)))
    if mae_1 > mae_2:
        mae_1 = mae_2
    cluster_mae.append(mae_1)
    globel_mae.append(mae_2)

print('Total')
counts = np.unique(train_cluster_label, return_counts=True)[1] / np.unique(train_cluster_label, return_counts=True)[1].sum()
print('clustering based:{:.2f}'.format(np.dot(cluster_mae, counts)))
print('global:{:.2f}'.format(np.dot(globel_mae, counts)))

label_counts = np.unique(train_cluster_label, return_counts=True)[1]