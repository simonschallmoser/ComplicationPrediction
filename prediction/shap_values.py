import numpy as np
import pandas as pd

import prediction

from joblib import Parallel, delayed
import shap

# Adjust these variables
random_seed = 8902
file_ending = 'basic1'

outcomes = ['eyes', 'nerves', 'renal', 'pvd', 'cevd', 'cavd']
param_grid = {
            'iterations': np.arange(50, 500),
            'learning_rate': np.logspace(-3, 0, 50),
            'depth': np.arange(3, 8),
             }

for population in ['diabetes', 'prediabetes']:
    data = np.load(f'/local/home/sschallmoser/complications/data/data_{population}_{file_ending}.npy', allow_pickle=True).flatten()[0] 
    Parallel(n_jobs=len(outcomes))(delayed(prediction.compute_shap_values)(outcome=outcome, 
                                                                           population=population,
                                                                           data=data[outcome],
                                                                           n_iter=20,
                                                                           n_splits=4,
                                                                           param_grid=param_grid,
                                                                           random_seed=random_seed,
                                                                           file_ending=file_ending)
              for outcome in outcomes)
