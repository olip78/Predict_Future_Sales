import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
import copy
import datetime
from itertools import product
import seaborn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
from random import sample
from sklearn.metrics import r2_score, mean_squared_error

def get_grid(main_data, shops_out={9, 20}):
    """
    shop/item/month grid for training data representation, repeating the test set structue
    """
    index_head = ['shop_id', 'item_id', 'date_block_num']
    data_for_training = []
    all_shops = set()
    all_items = set()
    for d in range(0, 35):
        d_shops = set(main_data[main_data['date_block_num']==d].shop_id.unique())-shops_out
        d_items = main_data[(main_data.date_block_num==d)&
                            (main_data.shop_id.isin(d_shops ))].item_id.unique()
        data_for_training.append(np.array(list(product(*[d_shops, d_items, [d]]))))
    data_for_training = pd.DataFrame(np.vstack(data_for_training), columns=index_head)
    data_for_training = data_for_training.sort_values(['date_block_num', 'shop_id', 'item_id'])
    return data_for_training.reset_index(drop=True)

class Sales_pt:
    """
    unstacked data representation for lagging and other manipulations with data
    """
    def __init__(self, main_data, index_keys, count=True, column='sales'):
        def f(x):
            try:
                n = list(x.values).index(1)
                return [0 if i < n else i + 1 - n for i in range(x.shape[0])]
            except ValueError:
                return list(np.zeros(x.shape[0], int))
        columns = ['date_block_num', column] + index_keys
        self.sum = main_data.loc[:, columns].pivot_table(index=index_keys, columns='date_block_num', 
                                                         values=column, aggfunc='sum') 
        if count:
            self.sales = main_data.loc[:, columns].pivot_table(index=index_keys, columns='date_block_num', 
                                                      values=column, aggfunc='count') 
            self.sales_np = np.array(list(self.sales.apply(f, axis=1).values))

def reduce_mem_usage(df, silent=True, allow_categorical=True, float_dtype="float32"):
    """ 
    Iterates through all the columns of a dataframe and downcasts the data type
     to reduce memory usage. Can also factorize categorical columns to integer dtype.
    author: Grant S 
    """
    def _downcast_numeric(series, allow_categorical=allow_categorical):
        """
        Downcast a numeric series into either the smallest possible int dtype or a specified float dtype.
        """
        if pd.api.types.is_sparse(series.dtype) is True:
            return series
        elif pd.api.types.is_numeric_dtype(series.dtype) is False:
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                return series
            else:
                if allow_categorical:
                    return series
                else:
                    codes, uniques = series.factorize()
                    series = pd.Series(data=codes, index=series.index)
                    series = _downcast_numeric(series)
                    return series
        else:
            series = pd.to_numeric(series, downcast="integer")
        if pd.api.types.is_float_dtype(series.dtype):
            series = series.astype(float_dtype)
        return series

    if silent is False:
        start_mem = np.sum(df.memory_usage()) / 1024 ** 2
        print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    if df.ndim == 1:
        df = _downcast_numeric(df)
    else:
        for col in df.columns:
            df.loc[:, col] = _downcast_numeric(df.loc[:,col])
    if silent is False:
        end_mem = np.sum(df.memory_usage()) / 1024 ** 2
        print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df

def to_vec(ss, sub_categories):
    res = ['']*3
    for i, k in enumerate(sub_categories.keys()):
        columns = sub_categories[k]
        for s in columns: 
            if s in ss: 
                res[i] = s
                continue
    return res

def kaggle_test_results(y):
    global data_for_training, target
    test = pd.read_csv(path.join(DATA_FOLDER, 'test.csv'))
    tmp = data_for_training[data_for_training['date_block_num']==34].loc[:, ['shop_id', 'item_id']]
    f = lambda x: np.clip(x, 0, 20)
    tmp.loc[:, 'item_cnt_month'] = f(y) 
    res =  test.merge(tmp, how = 'left', on=['shop_id', 'item_id']).loc[:, ['ID', 'item_cnt_month']]
    res.to_csv('results.csv', index=False)
    return res

class My_Methric:
    def __init__(self, y_train, y_test):
        self.f = np.vectorize(lambda x: np.clip(x, 0, 20))
        self.y_train = y_train
        self.y_test  = y_test
        self.current_best = {}
    def val(self, y, mode, method, columns=None):
        y_pred = self.f(y)
        if mode == 'test': y_true = self.y_test
        else: y_true = self.y_train    
        print('{} R-squared for {}: {:f}'.format(mode, method, r2_score(y_true, y_pred)))
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        print('{} RMSE for {}: {}'.format(mode, method, rmse)) 

def weights(XX, sonder=False):
    f = lambda x: -999 if x==0 else x
    X = XX.loc[:, 'sales_EMA6_x'].map(f)
    def f(x):
        if x > 1: return x
        else: return 1
    vfunc = np.vectorize(f, otypes=[np.float])
    if not sonder:
        res = vfunc(X.values)
    else:
        tmp = XX.loc[:, ['sonder_ladens', 'sonder_items']].sum(axis=1).map(lambda x: 1 + 10*x).values
        res = vfunc(X.values)*tmp
    return res

def get_features(keys):
    columns = []
    for b in keys: 
        columns = columns + final_features[b] 
    return columns    

def det_shap_sample(n, indices, target, type_ = 'sales'):
    df=pd.DataFrame(target.loc[indices, :])
    if   type_=='sales': ss = [0.3, 0.35, 0.35]
    elif type_=='regular sales': ss = [0.3, 0, 0.7]
    elif type_=='irregular sales': ss = [0.3, 0.7, 0]        
    
    new = list(df[df[type_]==0].index)
    res = sample(new, int(n*ss[0]))
    new = list(df[df[type_]>0].index)
    tmp = df.loc[new, type_].value_counts()
    s = len(new)
    for i in tmp.index:
        res = res + sample(new, int(ss[1]*n*tmp[i]/s))
    new = list(df[df[type_]>0].index)
    tmp = df.loc[new, type_].value_counts()
    s = len(new)
    for i in tmp.index:
        res = res + sample(new, int(ss[2]*n*tmp[i]/s))
    return res 

def make_data_for_training(to_concate):
    index_head = ['date_block_num', 'shop_id', 'item_id']
    data_for_training = data_for_training_base.loc[:, index_head]
    features = []
    for b in to_concate:
        data_for_training = pd.concat([data_for_training, b[0].loc[:, b[1]]], axis=1)
        features += b[1]
    return data_for_training, features

def get_train_set(data_for_training, features, sampler, t=33, t0=6, shop_filter=False):
    if shop_filter:
        ii = sorted(list(set(sampler.in_use)&set(data_for_training.query('shop_id in @shops_in & @t0<=date_block_num<@t').index)))
    else: ii = sorted(list(set(sampler.in_use)&set(data_for_training.query('@t0<=date_block_num<@t').index)))
    X_train = data_for_training.loc[ii, features]
    y_train = target.loc[ii, 'sales']
    ii = sorted(list(set(sampler.in_use)&set(data_for_training.query('date_block_num==@t').index)))
    X_test = data_for_training.loc[ii, features]
    if t < 34:
        y_test = target.loc[ii, 'sales']
    else: 
        y_test = []
    ii = sorted(list(set(sampler.in_use)&set(data_for_training.query('date_block_num==34').index)))   
    X_kaggle_test = data_for_training.loc[ii, features] 
    return X_train, X_test, y_train, y_test, X_kaggle_test

def add_knn_sgd(X, knn_sgd, choose_, till_m=33, train=True):
    columns = []
    for c in knn_sgd.columns:
        for b in choose_:
            if b in c:
                columns.append(c)
                break
    for c in ['sgd_modified_huber_0', 'sgd_modified_huber_20']:
        if not c in columns: columns.append(c) 
    if train:            
        X = pd.concat([X.fillna(-999).reset_index(drop=True), 
                       knn_sgd.query('date_block_num<@till_m').loc[:, columns].reset_index(drop=True)], axis=1)
    else:
        X = pd.concat([X.fillna(-999).reset_index(drop=True), 
                   knn_sgd.query('date_block_num==@till_m').loc[:, columns].reset_index(drop=True)], axis=1)
    return X
