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

outcomes = ['eyes', 'nerves', 'renal', 'pvd', 'cevd', 'cavd']

# Define variables
model = 'gru'
file_ending = 'lstm'
file_ending1 = 'basic'
type_ = 'all'
multitask = False
random_seed = 23
predictors = ['age', 'sex', 'bmi', 'systolic_bp', 'diastolic_bp', 'test=glucose','test=hba1c(%)',
              'test=creatinineserum']

for population in ['diabetes', 'prediabetes']:
    data = np.load(f'/local/home/sschallmoser/complications/data/data_{population}_{file_ending}.npy', allow_pickle=True).flatten()[0]
    for outcome in outcomes:
        prediction.prediction(data_final=data,
                              random_seed=random_seed,
                              model=model,
                              population=population,
                              outcome=outcome,
                              type_=type_,
                              file_ending1=file_ending1,
                              predictors=predictors, 
                              scale=False,
                              n_iter=20,
                              n_outer_splits=5,
                              n_inner_splits=4, 
                              multitask=multitask,
                              imputation_method=None)
