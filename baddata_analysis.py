import pandas as pd
import numpy as np

train = pd.read_csv('data/train/train.csv')

#%%
np.corrcoef(train['DHI'], train['TARGET'])
np.corrcoef(train['DNI'], train['TARGET'])
np.corrcoef(train['WS'], train['TARGET'])
np.corrcoef(train['RH'], train['TARGET'])
np.corrcoef(train['T'], train['TARGET'])

np.corrcoef(train['DHI'] + train['DNI'], train['TARGET'])

#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
train['TARGET'] = scaler.fit_transform(train['TARGET'].values.reshape(-1,1))
scaler = StandardScaler(with_mean=False)
train['D_sum'] = scaler.fit_transform(train['D_sum'].values.reshape(-1,1))
train['DHI'] = scaler.fit_transform(train['DHI'].values.reshape(-1,1))
train['DNI'] = scaler.transform(train['DNI'].values.reshape(-1,1))
plt.plot(train['TARGET'][-48*7:],label='Target')
# plt.plot(train['DHI'][-48:] )
# plt.plot(train['DNI'][-48:] )
plt.plot(train['D_sum'][-48*7:], label='D_sum')
plt.legend()
plt.show()