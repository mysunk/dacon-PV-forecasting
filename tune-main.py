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

def load_obj(name):
    with open('tune_results/' + name + '.pkl', 'rb') as f:
        trials = pickle.load(f)
        trials = sorted(trials, key=lambda k: k['loss'])
        return trials

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
    history = model.fit(X_train, y_train, epochs=params['EPOCH'], verbose=0, batch_size=params['BATCH_SIZE'],
                        validation_data=(X_val, y_val), callbacks=[es])
    # validation
    return np.min(history.history['val_loss'])

class Tuning_model(object):

    def __init__(self):
        self.random_state = 0
        self.space = {}

    # parameter setting
    def rf_space(self):
        self.space =  {
            'max_depth':                hp.quniform('max_depth',1, 20,1),
            'min_samples_leaf':         hp.quniform('min_samples_leaf', 1,10,1),
            'min_samples_split':        hp.uniform('min_samples_split', 0,1),
            'n_estimators':             1000,
            'max_features':             1,
            'criterion':                hp.choice('criterion', ['mse', 'mae']),
            'random_state' :            self.random_state,
            'n_jobs': -1
           }

    def dct_space(self):
        self.space = {
            'max_depth':                hp.quniform('max_depth', 2, 20, 1),
            'min_samples_leaf':         hp.quniform('min_samples_leaf', 1, 10, 1),
            'min_samples_split':        hp.uniform('min_samples_split', 0, 1),
            'criterion':                hp.choice('criterion', ['mse', 'mae']),
            }

    def dnn_space(self):
        self.space = {
            'EPOCH':                    1000,
            'BATCH_SIZE':               hp.quniform('BATCH_SIZE', 32, 512, 64),
            'h1':                       hp.quniform('h1', 48, 24*20, 48),
            'h2':                       hp.quniform('h2', 48, 24*20, 48),
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
            val = pinball(y_val, val_pred, q)
            val_results.append(val)

        # Dictionary with information for evaluation
        return {'loss': np.mean(val_results), 'params': params, 'status': STATUS_OK, 'method':args.method}

    def dnn_val(self, params, train_set):
        params = make_param_int(params, ['EPOCH', 'h1', 'h2', 'BATCH_SIZE'])
        train_data, train_label = train_set
        X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size = 0.2, random_state = 42)

        # pre-precessing
        X_train = X_train.transpose((0, 2, 1))
        X_val = X_val.transpose((0, 2, 1))
        X_train = X_train.reshape(-1, past_history * FEATURES)
        X_val = X_val.reshape(-1, past_history * FEATURES)

        val_results = []
        for i, q in enumerate(np.arange(0.1, 1, 0.1)):
            g = tf.Graph()
            with g.as_default():
                val = dnn_val(X_train, y_train, X_val, y_val, params, q)
                val_results.append(val)
            tf.keras.backend.clear_session()
            break

        # Dictionary with information for evaluation
        return {'loss': np.mean(val_results), 'params': params, 'status': STATUS_OK, 'method':args.method}


if __name__ == '__main__':

    # load config
    parser = argparse.ArgumentParser(description='PV forecasting',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method', default='dnn', choices=['rf','dnn'])
    parser.add_argument('--max_evals', default=1, type=int)
    parser.add_argument('--lags', default=3, type=int)
    parser.add_argument('--save_file', default='tmp')
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

    x_col = ['DHI', 'DNI', 'D_sum', 'WS', 'RH', 'T', 'TARGET']
    y_col = ['TARGET']

    train = pd.read_csv('data/train/train.csv')
    submission = pd.read_csv('data/sample_submission.csv')
    submission.set_index('id', inplace=True)
    train['D_sum'] = train['DHI'] + train['DNI']

    dataset = train.loc[:, x_col].values
    label = np.ravel(train.loc[:, y_col].values)

    FEATURES = len(x_col)
    past_history = 48 * args.lags
    future_target = 48 * 2

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