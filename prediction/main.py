import numpy as np
import pandas as pd

from joblib import Parallel, delayed

import prediction 

outcomes = ['eyes', 'nerves', 'renal', 'pvd', 'cevd', 'cavd']
random_seed = 23
path = '/local/home/sschallmoser/complications/data/'

for population in ['diabetes', 'prediabetes']:
    for model in ['logreg', 'catboost']:        
        file_endings = ['basic', 'trajectory']
        for file_ending in file_endings:
            file_ending1 = file_ending
            if file_ending == 'trajectory':
                data = np.load(path+f'data_{population}_{file_ending}.npy', allow_pickle=True).flatten()[0]
            else:
                data = np.load(path+f'data_{population}_{file_ending}1.npy', allow_pickle=True).flatten()[0]

            Parallel(n_jobs=len(outcomes))(delayed(prediction.prediction)(data_final=data,
                                                                          random_seed=random_seed,
                                                                          model=model,
                                                                          population=population,
                                                                          outcome=outcome,
                                                                          type_='all',
                                                                          file_ending1=file_ending1,
                                                                          predictors=None, 
                                                                          scale=False,
                                                                          n_iter=20,
                                                                          n_outer_splits=5,
                                                                          n_inner_splits=4, 
                                                                          multitask=False,
                                                                          imputation_method=None)
                                               for outcome in outcomes)
        predictors = ['age', 'sex', 'bmi', 'systolic_bp', 'diastolic_bp', 'test=glucose','test=hba1c(%)',
                      'test=creatinineserum']
        file_ending = 'basic'
        data = np.load(path+f'data_{population}_{file_ending}1.npy', allow_pickle=True).flatten()[0]
        Parallel(n_jobs=len(outcomes))(delayed(prediction.prediction)(data_final=data,
                                                                      random_seed=random_seed,
                                                                      model=model,
                                                                      population=population,
                                                                      outcome=outcome,
                                                                      type_='all',
                                                                      file_ending1='trajectory_predictors',
                                                                      predictors=predictors, 
                                                                      scale=False,
                                                                      n_iter=20,
                                                                      n_outer_splits=5,
                                                                      n_inner_splits=4, 
                                                                      multitask=False,
                                                                      imputation_method=None)
                                           for outcome in outcomes)
