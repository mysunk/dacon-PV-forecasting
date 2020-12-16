import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pinball_loss(q,y_true, y_pred):
    idx1 = y_true >= y_pred
    idx2 = y_true < y_pred

    y_true_1, y_pred_1 = np.ravel(y_true[idx1]),np.ravel(y_pred[idx1])
    y_true_2, y_pred_2 = np.ravel(y_true[idx2]),np.ravel(y_pred[idx2])

    loss_1 = (y_true_1 - y_pred_1)*q
    loss_2 = (y_pred_2 - y_true_2) * (1 - q)

    loss = np.concatenate([loss_1, loss_2])
    return np.mean(loss)


val_preds_df = pd.read_csv('val/val_11.csv')
val_preds_df.set_index('Unnamed: 0',inplace=True)

# print validation pinball loss
val_results = []
for q in np.arange(0.1,1,0.1):
    val = pinball_loss(q,val_preds_df['true'], val_preds_df['q_'+str(q)[:3]])
    val_results.append(val)
print(np.mean(val_results, axis=0))

# plot sample prediction
plt.plot(val_preds_df.iloc[48:48*2,9],'r',label='True')
for i in range(9):
    plt.plot(val_preds_df.iloc[48:48*2,i],'x',label=val_preds_df.columns[i])
plt.legend()
plt.show()