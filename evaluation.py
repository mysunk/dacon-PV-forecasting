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


val_preds_df = pd.read_csv('val/val_4.csv')
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

#%%
submit_21 = pd.read_csv('submit/submit_0.csv')
submit_24 = pd.read_csv('submit/submit_25.csv')
result = 0
for i in range(1,10):
    result += pinball_loss(0.1 * i, submit_21['q_0.1'].values, submit_24['q_0.1'].values)
result /= 9
print(result)

#%%
submit_1 = pd.read_csv('submit/submit_4.csv')
submit_2 = pd.read_csv('submit/submit_16.csv')
submit_1.iloc[:,1:] = ((submit_1.iloc[:,1:] + submit_2.iloc[:,1:])/2).values.copy()
submit_1.to_csv('submit/submit_0.csv', index=False)
