def trans(dataset, pasts, future, x_col_index, y_col_index, STEP=48):
    pasts_rev = np.insert(pasts+1, 0, 0)
    n_rows = dataset.shape[0]-pasts.max()-future+STEP
    n_rows = n_rows // STEP
    data_agg = np.zeros((n_rows,pasts.sum()+len(pasts)))
    labels = np.zeros((n_rows, future))
    start, end = 0, dataset.shape[0]
    for j, x_col in enumerate(x_col_index):
        start = start + pasts[j]
        data = []
        dataset_sub = dataset[:, x_col]
        for i in range(start, end-future, STEP):
            indices = np.array(dataset_sub[i - pasts[j]:i+1])
            data.append(indices)
        data = np.array(data)
        data = data.reshape(data.shape[0], -1)
        data = data[(max(pasts) - pasts[j])//STEP:, :]
        data_agg[:,pasts_rev[:j+1].sum():pasts_rev[:j+2].sum()] = data
        start = 0
    for j, i in enumerate(range(max(pasts), end - future,STEP)):
        labels[j,:] = np.array(dataset[i+1:i + future+1, y_col_index])
    return data_agg, labels

x_columns = ['DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET']
y_column = 'TARGET'

# history and future length
past = 48*2 - 1 # 이틀
future = 48*2 # 이틀
pasts = (np.ones(len(x_columns)) * past).astype(int)

# column to idx
x_col_index = np.zeros(np.shape(x_columns),dtype=int)
for i, x_column in enumerate(x_columns):
    x_col_index[i] = np.where(x_column == train.columns)[0][0]
y_col_index = np.where(y_column == train.columns)[0][0]

# transform
X, y = trans(train.values, pasts, future, x_col_index, y_col_index)

#%% train-test split
TRAIN_SPLIT = int(X.shape[0] * 0.8)
X_train, y_train = X[:TRAIN_SPLIT,:], y[:TRAIN_SPLIT,:]
X_val, y_val = X[TRAIN_SPLIT:,:], y[TRAIN_SPLIT:,:]