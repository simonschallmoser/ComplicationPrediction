import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from importlib import reload
import sys 
sys.path.insert(0, '../preprocessing')

from joblib import Parallel, delayed
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from scipy.stats import randint, uniform

from data_reader import read_demo_data, read_lab_data, read_icd9_data, read_med_data
import helper
import preprocess
import data_imputation
import prediction 

outcomes = ['eyes', 'renal', 'nerves', 'pvd', 'cavd', 'cevd']

# Define variables
model = 'catboost'
file_ending = 'basic'
type_ = 'all'
random_seed = 23

for population in ['diabetes', 'prediabetes']:
    data = np.load(f'/local/home/sschallmoser/complications/data/data_{population}_{file_ending}1.npy', allow_pickle=True).flatten()[0]
    
    ids = []
    for outcome in outcomes:
        ids.append(set(list(data[outcome].index.get_level_values('id'))))
    ids_all = list(ids[0].intersection(ids[1]).intersection(ids[2]).intersection(ids[3]).intersection(ids[4]).intersection(ids[5]))
    data_all = data['eyes'].loc[ids_all].drop('y', axis=1).copy()
    for outcome in outcomes:
        data_all[outcome] = data[outcome].loc[ids_all]['y']
    data_all['any'] = data_all[outcomes].sum(axis=1).astype(bool).astype(int)
    
    data_healthy = {}
    for outcome in outcomes:
        data_sub = data_all.copy()
        data_sub['y'] = data_sub[outcome]
        data_sub = data_sub.drop(np.append(outcomes, 'any'), axis=1)
        data_healthy[outcome] = data_sub
    
    for outcomes_, file_ending1, multitask in zip([outcomes, outcomes, ['any']], ['healthy', 'multitask', 'real_multitask'], [False, 'multitask', 'real_multitask']):
        print(outcomes_, file_ending1, multitask)
        if file_ending1 == 'healthy':
            data_ = data_healthy.copy()
        else:
            data_ = data_all.copy()
        Parallel(n_jobs=len(outcomes_))(delayed(prediction.prediction)(data_final=data_,
                                                                       random_seed=random_seed,
                                                                       model=model,
                                                                       population=population,
                                                                       outcome=outcome,
                                                                       type_=type_,
                                                                       file_ending1=file_ending1,
                                                                       predictors=None, 
                                                                       scale=False,
                                                                       n_iter=20,
                                                                       n_outer_splits=5,
                                                                       n_inner_splits=4, 
                                                                       multitask=multitask,
                                                                       imputation_method=None)
                for outcome in outcomes_)
