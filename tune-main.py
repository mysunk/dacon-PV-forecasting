import argparse
import numpy as np
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, STATUS_FAIL
from functools import partial
import pandas as pd
# models
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import os
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

# GPU setting
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

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

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pickle
def save_obj(obj, name):
    with open('tune_results/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def make_param_int(param, key_names):
    for key, value in param.items():
        if key in key_names:
            param[key] = int(param[key])
    return param

def pinball(y_true, y_pred, q):
    pin = K.mean(K.maximum(y_true - y_pred, 0) * q +
                 K.maximum(y_pred - y_true, 0) * (1 - q))
    return pin

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
    model.add(tf.keras.layers.Dense(params['h3'], activation='relu'))
    model.add(tf.keras.layers.Dense(params['h4'], activation='relu'))
    model.add(tf.keras.layers.Dense(len(idx)))

    optimizer = tf.keras.optimizers.Adam(params['lr'])
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=0,
        mode='auto',
        restore_best_weights=True
    )
    model.compile(optimizer=optimizer, loss=custom_loss(q=q), metrics=[custom_loss(q=q)])
    history = model.fit(X_train, y_train, epochs=params['EPOCH'], verbose=0, batch_size=params['BATCH_SIZE'],
                        validation_data=(X_val, y_val), callbacks=[es])
    val_pred = np.zeros((y_val.shape[0],96))
    val_pred[:, idx] = model.predict(X_val)
    return np.min(history.history['val_loss']), val_pred, model

class Tuning_model(object):

    def __init__(self):
        self.random_state = 0
        self.space = {}

    # parameter setting
    def rf_space(self):
        self.space =  {
            'max_depth':                hp.quniform('max_depth',1, 10,1),
            'min_samples_leaf':         hp.quniform('min_samples_leaf', 1,10,1),
            'min_samples_split':        hp.uniform('min_samples_split', 0,1),
            'n_estimators':             1000,
            'max_features':             1,
            'criterion':                hp.choice('criterion', ['mse', 'mae']),
            'random_state' :            self.random_state,
            'n_jobs': -1
           }

    def extra_space(self):
        self.space = {
            'max_depth':                hp.quniform('max_depth', 2, 20, 1),
            'min_samples_leaf':         hp.quniform('min_samples_leaf', 1, 10, 1),
            'min_samples_split':        hp.uniform('min_samples_split', 0, 1),
            'criterion':                hp.choice('criterion', ['mse', 'mae']),
            'max_features':             1,
            'n_jobs':                   -1,
            'n_estimators':             1000,
            'random_state':             self.random_state,
            }

    def dnn_space(self):
        self.space = {
            'EPOCH':                    1000,
            'BATCH_SIZE':               hp.quniform('BATCH_SIZE', 32, 512, 64),
            'h1':                       hp.quniform('h1', 48, 24*20, 48),
            'h2':                       hp.quniform('h2', 48, 24*20, 48),
            'h3':                       hp.quniform('h3', 48, 24 * 20, 48),
            'h4':                       hp.quniform('h4', 48, 24 * 20, 48),
            'lr':                       hp.loguniform('lr',np.log(1e-4),np.log(1e-1))
            }

    # optimize
    def process(self, clf_name, train_set, trials, algo, max_evals):
        fn = getattr(self, clf_name+'_val')
        space = getattr(self, clf_name+'_space')
        space()
        fmin_objective = partial(fn, train_set=train_set)
        try:
            result = fmin(fn=fmin_objective, space=self.space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def rf_val(self, params, train_set):
        params = make_param_int(params, ['max_depth', 'max_features', 'n_estimators', 'min_samples_leaf'])
        train_data, train_label = train_set
        X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size = 0.2, random_state = 42)

        # pre-precessing
        X_train = X_train.transpose((0, 2, 1))
        X_val = X_val.transpose((0, 2, 1))
        X_train = X_train.reshape(-1, past_history * FEATURES)
        X_val = X_val.reshape(-1, past_history * FEATURES)

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        rf_preds = []
        for estimator in rf.estimators_:
            rf_preds.append(estimator.predict(X_val))
        rf_preds = np.array(rf_preds)

        val_results = []
        for i, q in enumerate(np.arange(0.1, 1, 0.1)):
            val_pred = np.percentile(rf_preds, q * 100, axis=0)
            if args.normalize:
                val = pinball(denormalize(y_val), denormalize(val_pred), q)
            else:
                val = pinball(y_val, val_pred, q)
            val_results.append(val)

        # Dictionary with information for evaluation
        return {'loss': np.mean(val_results), 'params': params, 'status': STATUS_OK, 'method':args.method}

    def extra_val(self, params, train_set):
        params = make_param_int(params, ['max_depth', 'max_features', 'n_estimators', 'min_samples_leaf'])
        train_data, train_label = train_set
        X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size = 0.2, random_state = 42)

        # pre-precessing
        X_train = X_train.transpose((0, 2, 1))
        X_val = X_val.transpose((0, 2, 1))
        X_train = X_train.reshape(-1, past_history * FEATURES)
        X_val = X_val.reshape(-1, past_history * FEATURES)

        regr = ExtraTreesRegressor(**params)
        regr.fit(X_train, y_train)

        extra_preds = []
        for estimator in regr.estimators_:
            extra_preds.append(estimator.predict(X_val))
        extra_preds = np.array(extra_preds)

        val_results = []
        for i, q in enumerate(np.arange(0.1, 1, 0.1)):
            val_pred = np.percentile(extra_preds, q * 100, axis=0)
            if args.normalize:
                val = pinball(denormalize(y_val), denormalize(val_pred), q)
            else:
                val = pinball(y_val, val_pred, q)
            val_results.append(val)

        # Dictionary with information for evaluation
        return {'loss': np.mean(val_results), 'params': params, 'status': STATUS_OK, 'method':args.method}

    def dnn_val(self, params, train_set):
        params = make_param_int(params, ['EPOCH', 'h1', 'h2','h3','h4', 'BATCH_SIZE'])
        train_data, train_label = train_set
        X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size = 0.2, random_state = 42)

        # pre-precessing
        X_train = X_train.transpose((0, 2, 1))
        X_val = X_val.transpose((0, 2, 1))
        X_train = X_train.reshape(-1, past_history * FEATURES)
        X_val = X_val.reshape(-1, past_history * FEATURES)

        val_preds = []
        for i, q in enumerate(np.arange(0.1, 1, 0.1)):
            g = tf.Graph()
            with g.as_default():
                _, val_pred, _ = dnn_val(X_train, y_train[:, idx], X_val, y_val[:, idx], params, q)
                if args.normalize:
                    val_pred = denormalize(val_pred)
                    if i==0:
                        y_val = denormalize(y_val)
                val_preds.append(val_pred)
            tf.keras.backend.clear_session()

        val_results = []
        for i, q in enumerate(np.arange(0.1, 1, 0.1)):
            loss = pinball(y_val, val_preds[i], q)
            val_results.append(loss)

        # Dictionary with information for evaluation
        return {'loss': np.mean(val_results), 'params': params, 'status': STATUS_OK, 'method':args.method}

if __name__ == '__main__':

    # load config
    parser = argparse.ArgumentParser(description='PV forecasting',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', default='extra', choices=['rf','dnn','extra'])
    parser.add_argument('--max_evals', default=1, type=int)
    parser.add_argument('--lags', default=2, type=int)
    parser.add_argument('--save_file', default='tmp')
    parser.add_argument('--normalize', default=True, type=bool)
    args = parser.parse_args()

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

    def normalize(df):
        df_n = (df - train_min) / (train_max - train_min)
        return df_n

    def denormalize(arr):
        arr = arr * (train_max['TARGET'] - train_min['TARGET']) + train_min['TARGET']
        return arr

    x_col = ['TARGET']
    y_col = ['TARGET']

    train = pd.read_csv('data/train/train.csv')
    submission = pd.read_csv('data/sample_submission.csv')
    submission.set_index('id', inplace=True)

    if args.normalize:
        train_max = train.max(axis=0)
        train_min = train.min(axis=0)
        train = normalize(train)

    dataset = train.loc[:, x_col].values
    label = np.ravel(train.loc[:, y_col].values)

    FEATURES = len(x_col)
    past_history = 48 * args.lags
    future_target = 48 * 2
    # used index
    idx = list(range(10, 39)) + list(range(10 + 48, 39 + 48))

    ### transform train
    train_data, train_label = multivariate_data(dataset, label, 0,
                                                None, past_history,
                                                future_target, 1,
                                                single_step=False)

    # main
    clf = args.method
    bayes_trials = Trials()
    obj = Tuning_model()
    tuning_algo = tpe.suggest # -- bayesian opt
    # tuning_algo = tpe.rand.suggest # -- random search
    obj.process(args.method, [train_data, train_label],
                           bayes_trials, tuning_algo, args.max_evals)

    # save trial
    save_obj(bayes_trials.results,args.save_file)