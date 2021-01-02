import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

train = pd.read_csv('data/train/train.csv')
train['D_sum'] = train['DHI'] + train['DNI']
submission = pd.read_csv('data/sample_submission.csv')
submission.set_index('id',inplace=True)

#%% preprocessing for train data
# 0. normalization
max_ = train.max(axis=0)
min_ = train.min(axis=0)
X_std = (train - min_) / (max_ - min_)
X_scaled = X_std * (max_ - min_) + min_
scaling_factors = [min_, max_]

# 1. feature extraction
features = ['DHI','DNI','D_sum','WS','RH','T']
data_w = []
for feature in features:
    data = X_std[feature].values
    data = data.reshape(-1,48)
    data_w.append(data.mean(axis=1))
    data_w.append(data.max(axis=1))
    data_w.append(data.min(axis=1))

data_w = np.array(data_w).T
target = X_std['TARGET'].values.reshape(-1,48)

# 2. fit weather model
from sklearn.ensemble import RandomForestRegressor
X = data_w[:-1,:]
y = data_w[1:,:]
rf_param = {'criterion':'mae', 'n_jobs': -1, 'max_depth':5, 'n_estimators': 100}
rf = RandomForestRegressor(**rf_param)
rf.fit(X, y)

# 3. fit clustering
from sklearn.cluster import KMeans
k = 3
kmeans = KMeans(n_clusters = k, random_state=0).fit(data_w)
cluster_label = kmeans.predict(data_w)

#%% test load and assign cluster label
### transform test
test_data = []
test_cluster_label = []
for i in range(81):
    data = []
    test = pd.read_csv(f'data/test/{i}.csv')
    test['D_sum'] = test['DHI'] + test['DNI']

    test_std = (test - min_) / (max_ - min_)

    test_data_w = []
    for feature in features:
        data = test_std[feature].values
        data = data.reshape(-1, 48)
        test_data_w.append(data.mean(axis=1))
        test_data_w.append(data.max(axis=1))
        test_data_w.append(data.min(axis=1))
    test_data_w = np.array(test_data_w).T
    predicted_label = kmeans.predict(test_data_w)

    next_day_w = rf.predict(test_data_w[-1:,:])
    next_day_label = kmeans.predict(next_day_w)[0]

    if next_day_label in predicted_label:
        idx = np.where(predicted_label == next_day_label)[0][-1]
        test_cluster_label.append(next_day_label)
    else:
        print('No matched cluster label')
        idx = 6 # last day
        test_cluster_label.append(predicted_label[idx])
    test_data.append(test_std['TARGET'].values[idx * 48:(idx + 1) * 48])

#%% fit with corresponding cluster labels
rf_models = []
for i in range(3):
    target_clustered = target[cluster_label == i]
    X_train = target_clustered[:-1,:]
    y_train = target_clustered[1:, :]
    rf_param = {'criterion': 'mae', 'n_jobs': -1, 'max_depth': 5, 'n_estimators': 1000,
                'max_features':1}
    rf = RandomForestRegressor(**rf_param)
    rf.fit(X_train, y_train)
    rf_models.append(rf)

#%% predict for test sets
for test_idx in tqdm(range(81)):
    data = test_data[test_idx].reshape(1,-1)
    label = test_cluster_label[test_idx]

    rf = rf_models[label]
    rf_preds = []
    for estimator in rf.estimators_:
        rf_preds.append(estimator.predict(data))
    rf_preds = np.array(rf_preds)

    for i, q in enumerate(np.arange(0.1, 1, 0.1)):
        y_pred = np.percentile(rf_preds, q * 100, axis=0)
        # inverse normalize
        y_pred = y_pred * (max_['TARGET'] - min_['TARGET']) + min_['TARGET']
        submission.iloc[test_idx*48:(test_idx+1)*48, i] = y_pred

submission.to_csv('submit/submit_22.csv')