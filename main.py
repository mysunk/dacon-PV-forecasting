import pandas as pd
import numpy as np

train = pd.read_csv('data/train/train.csv')
test_sample = pd.read_csv('data/test/0.csv')
submission = pd.read_csv('data/sample_submission.csv')
submission.set_index('id',inplace=True)

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

used_features =['TARGET']
dataset = train.loc[:,used_features].values
FEATURES = len(used_features)
past_history = 48*3
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
    tmp = pd.read_csv(f'data/test/{i}.csv')
    tmp = tmp.loc[:, used_features].values
    tmp = tmp[-past_history:,:]
    tmp = tmp.reshape(-1, past_history, FEATURES)
    test.append(tmp)
test = np.concatenate(test, axis=0)

#%% train
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import initializers
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam

def custom_loss(q = 0.1):
    def pinball_loss(y_true, y_pred):
        idx1 = y_true >= y_pred
        idx2 = y_true < y_pred

        y_true_1, y_pred_1 = K.flatten(y_true[idx1]),K.flatten(y_pred[idx1])
        y_true_2, y_pred_2 = K.flatten(y_true[idx2]),K.flatten(y_pred[idx2])

        loss_1 = (y_true_1-y_pred_1)*q
        loss_2 = (y_pred_2 - y_true_2) * (1 - q)

        loss = K.concatenate([loss_1, loss_2])
        return K.mean(loss)
    return pinball_loss

# model train
model = Sequential([
    tf.keras.layers.LSTM(past_history,
                         return_sequences=False,
                         input_shape=X_train.shape[-2:],
                         kernel_initializer=initializers.he_normal()),
    tf.keras.layers.Dropout(.1),
    tf.keras.layers.Dense(future_target*3,
                          kernel_initializer=initializers.he_normal()),
    tf.keras.layers.Dropout(.1),
    tf.keras.layers.Dense(future_target,
                          kernel_initializer=initializers.he_normal()),
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

#%% train model
EPOCHS = 100
BATCH_SIZES = 256

val_preds_df = pd.DataFrame(columns=list(submission.columns) + ['true'],
                           data=np.zeros((y_val.shape[0] * future_target,10)))
val_preds_df['true'] = np.ravel(y_val)
optimizer = Adam(learning_rate=0.01)

for i, q in enumerate(np.arange(0.1, 1, 0.1)):
    model.compile(optimizer=optimizer, loss=custom_loss(q=q), metrics=[custom_loss(q=q)])
    history = model.fit(X_train, y_train, epochs=EPOCHS,validation_data=(X_val, y_val), batch_size=BATCH_SIZES,
                        callbacks=[lr, es], verbose=2)
    tf.keras.backend.clear_session()

    # validation
    val_pred = model.predict(X_val)
    val_preds_df.iloc[:,i] = np.ravel(val_pred)

    # test pred
    y_pred = model.predict(test)
    submission.iloc[:,i] = np.ravel(y_pred)

#%% evaluation
def pinball_loss(y_true, y_pred, q=0.5):
    idx1 = y_true >= y_pred
    idx2 = y_true < y_pred

    y_true_1, y_pred_1 = np.ravel(y_true[idx1]),np.ravel(y_pred[idx1])
    y_true_2, y_pred_2 = np.ravel(y_true[idx2]),np.ravel(y_pred[idx2])

    loss_1 = (y_true_1 - y_pred_1)*q
    loss_2 = (y_pred_2 - y_true_2) * (1 - q)

    loss = np.concatenate([loss_1, loss_2])
    return np.mean(loss)

val_results = []
val_preds_df[val_preds_df<0] = 0
for q in np.arange(0.1,1,0.1):
    val = pinball_loss(val_preds_df['true'], val_preds_df['q_'+str(q)[:3]], q=q)
    val_results.append(val)
print(np.mean(val_results, axis=0))

#%% minus to 0
submission[submission<0] = 0
submission.to_csv('submit/submit_1.csv')
# save corresponding val result
val_preds_df.to_csv('val/val_1.csv')