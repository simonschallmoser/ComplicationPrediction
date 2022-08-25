import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import sys
sys.path.insert(0, '../preprocessing')
import warnings
warnings.filterwarnings("ignore")

from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV, ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

import data_reader
import helper
import preprocess
import data_imputation
import shap

def compute_metrics(y_test,
                    y_pred,
                    model,
                    pipe):
    auroc = metrics.roc_auc_score(y_test, y_pred)
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
    auprc = metrics.auc(recall, precision)
    fpr, tpr, thr = metrics.roc_curve(y_test, y_pred)
    tpr_opt = tpr[np.argmax(tpr-fpr)]
    tnr_opt = 1 - fpr[np.argmax(tpr-fpr)]
    thr_opt = thr[np.argmax(tpr-fpr)]
    y_pred_rounded = [1 if i > thr_opt else 0 for i in y_pred]
    pr = np.round(np.count_nonzero(y_pred_rounded)/len(y_pred_rounded)*100, 1)
    prec_opt = metrics.precision_score(y_test, y_pred_rounded)
    bacc_opt = metrics.balanced_accuracy_score(y_test, y_pred_rounded)
    if model == 'logreg':
        coefs = pipe['model'].coef_[0]
    else:
        coefs = None
    
    return auroc, auprc, tpr_opt, tnr_opt, prec_opt, bacc_opt, pr, coefs

def unpack_metrics(all_metrics: list,
                   outcome,
                   population,
                   model,
                   n_bootstrapping_iterations,
                   file_ending2):
    if n_bootstrapping_iterations != 0:
        method = 'bootstrap'
        lower = 2.5
        upper = 97.5
    else:
        method = 'range'
        lower = 0
        upper = 100
    aurocs_all = [all_metrics[i][0] for i in range(len(all_metrics))]
    aurocs = (np.mean(aurocs_all), np.percentile(aurocs_all, lower), np.percentile(aurocs_all, upper))    
    auprcs = [all_metrics[i][1] for i in range(len(all_metrics))]
    auprcs = (np.mean(auprcs), np.percentile(auprcs, lower), np.percentile(auprcs, upper))
    sensitivities = [all_metrics[i][2] for i in range(len(all_metrics))]
    sensitivities = (np.mean(sensitivities), np.percentile(sensitivities, lower), np.percentile(sensitivities, upper))
    specificities = [all_metrics[i][3] for i in range(len(all_metrics))]
    specificities = (np.mean(specificities), np.percentile(specificities, lower), np.percentile(specificities, upper))
    precisions = [all_metrics[i][4] for i in range(len(all_metrics))]
    precisions = (np.mean(precisions), np.percentile(precisions, lower), np.percentile(precisions, upper))
    balanced_accuracies = [all_metrics[i][5] for i in range(len(all_metrics))]
    balanced_accuracies = (np.mean(balanced_accuracies), np.percentile(balanced_accuracies, lower),
                           np.percentile(balanced_accuracies, upper))
    positivity_rates = [all_metrics[i][6] for i in range(len(all_metrics))]
    positivity_rates = (np.mean(positivity_rates), np.percentile(positivity_rates, lower),
                        np.percentile(positivity_rates, upper))
    if model == 'logreg':
        coefs = [all_metrics[i][7] for i in range(len(all_metrics))]
    else:
        coefs = None
    results = {'auroc': aurocs, 
               'aurocs_all': aurocs_all, 
               'auprc': auprcs, 
               'tpr': sensitivities, 
               'tnr': specificities, 
               'precision': precisions, 
               'bacc': balanced_accuracies, 
               'pr': positivity_rates,
               'coefs': coefs}
    np.save(f'results/results_{population}_{outcome}_{model}_{method}_{file_ending2}.npy', results)
    
    return aurocs, aurocs_all, auprcs, sensitivities, specificities, precisions, balanced_accuracies, positivity_rates  

def unpack_metrics_mean(all_metrics: list):

    auroc = np.mean([all_metrics[i][0] for i in range(len(all_metrics))])
    auprc = np.mean([all_metrics[i][1] for i in range(len(all_metrics))])
    sensitivity = np.mean([all_metrics[i][2] for i in range(len(all_metrics))])
    specificity = np.mean([all_metrics[i][3] for i in range(len(all_metrics))])
    precision = np.mean([all_metrics[i][4] for i in range(len(all_metrics))])
    balanced_accuracy = np.mean([all_metrics[i][5] for i in range(len(all_metrics))])
    positivity_rate = np.mean([all_metrics[i][6] for i in range(len(all_metrics))])
    coefs = [all_metrics[i][7] for i in range(len(all_metrics))]
    
    return auroc, auprc, sensitivity, specificity, precision, balanced_accuracy, positivity_rate, coefs

class StandardScaler1:
    
    def __init__(self, copy=True):
        self.copy=copy    
    
    def fit(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.len_time = int(len(X) / len(X.index.get_level_values('id').unique()))
        self.len_columns = len(X.columns)
        
        return self
    
    def fit_transform(self, X, y=None):
        len_patients = len(X.index.get_level_values('id').unique())
        self = self.fit(X)
        X_new = np.divide(X - self.means, self.stds)
        
        return np.array(X_new).reshape((len_patients, self.len_time, self.len_columns))
    
    def transform(self, X, y=None):
        len_patients = len(X.index.get_level_values('id').unique())
        X_new = np.divide(X - self.means, self.stds)
        
        return np.array(X_new).reshape((len_patients, self.len_time, self.len_columns))
    
def initialize_model(model, 
                     multitask=False):
    if model == 'logreg':
        clf = LogisticRegression(solver='saga', n_jobs=1)
    elif model == 'catboost':
        if multitask:
            clf = CatBoostClassifier(thread_count=1, verbose=0, loss_function='MultiCrossEntropy')
        else:
            clf = CatBoostClassifier(thread_count=1, verbose=0)
    elif model == 'lgbm':
        clf = LGBMClassifier(n_jobs=1)
    return clf

def rnn(model_type, len_time, len_columns, n_cells, dropout_rate, recurrent_dropout_rate, learning_rate):
    """
    Initialize LSTM.
    """
    
    keras_model = Sequential()
    if model_type == 'lstm':
        keras_model.add(LSTM(n_cells, input_shape=(len_time, len_columns), dropout=dropout_rate, 
                             recurrent_dropout=recurrent_dropout_rate))
    elif model_type == 'gru':
        keras_model.add(GRU(n_cells, input_shape=(len_time, len_columns), dropout=dropout_rate,
                            recurrent_dropout=recurrent_dropout_rate))
    keras_model.add(Dense(1, activation='sigmoid'))
    keras_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate))
    
    return keras_model

def prediction(data_final,
               random_seed,
               model,
               population,
               outcome,
               type_,
               file_ending1,
               predictors=None, 
               scale=False,
               n_iter=20,
               n_outer_splits=5,
               n_inner_splits=4, 
               multitask=False,
               imputation_method=None,
               store=True):
    """
    """      
    # Specify data
    if not multitask:
        data_final = data_final[outcome]
        
    # Specfiy included columns depening on type_
    demo_cols = ['age', 'bmi', 'sex', 'systolic_bp', 'diastolic_bp']
    lab_cols = [i for i in data_final.columns if 'test' in i]
    icd9_cols = [i for i in data_final.columns if 'icd' in i]
    med_cols = [i for i in data_final.columns if 'med' in i]
    if type_ == 'all' and predictors is None:
        predictors = None
    elif type_ == 'demo':
        predictors = demo_cols
    elif type_ == 'lab':
        predictors = np.append(demo_cols, lab_cols)
    elif type_ == 'icd':
        predictors = np.append(demo_cols, icd9_cols)
    elif type_ == 'med':
        predictors = np.append(demo_cols, med_cols)
    
    # Specify parameters for logistic regression
    if model == 'logreg':
        scale = True
        imputation_method = 'median'
        param_grid = {
                     'penalty': ['l1'],
                     'C': np.logspace(-3, 3, 50),
                     }
    elif model == 'lstm' or model == 'gru':
        param_grid = {
                     'n_cells': [3, 5, 8, 10, 15, 50],
                     'dropout_rate': [0, 0.2, 0.5],
                     'recurrent_dropout_rate': [0, 0.2, 0.5],
                     'learning_rate': [0.0001, 0.001, 0.01, 0.1],
                     'batch_size': [64, 128],
                     'epochs': np.arange(5, 30)
                     }
    else:
        param_grid = {
                     'iterations': np.arange(50, 500),
                     'learning_rate': np.logspace(-3, 0, 50),
                     'depth': np.arange(3, 8),
                      }
    # Define X and y
    if multitask == 'real_multitask' or multitask == 'multitask':
        X = data_final.drop(['eyes', 'renal', 'nerves', 'pvd', 'cevd', 'cavd', 'any'], axis=1)
    else:
        X = data_final.drop('y', axis=1)
        
    # If predictors are specified, select only those
    if predictors is not None:
        X = X[predictors]
    #print(X.columns)
    if multitask == 'real_multitask':
        y = data_final[['eyes', 'renal', 'nerves', 'pvd', 'cevd', 'cavd', 'any']]
    elif multitask == 'multitask':
        y = data_final[[outcome, 'any']]
    else:
        y = data_final['y']
        y = y.dropna()
    #print('No. of outcomes:', np.sum(y))
    #if np.sum(y) == 0:
    #    raise ValueError('No outcomes.')
    
    # Model prediction using stratified cross-validation
    y_pred_all = []
    y_test_all = []
    if multitask == 'real_multitask':
        skf_outer = KFold(n_splits=n_outer_splits, shuffle=True, random_state=random_seed)
    else: 
        skf_outer = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=random_seed)
    if multitask == 'multitask':
        y_split = y[outcome]
    else:
        y_split = y
    for train_index, test_index in skf_outer.split(np.zeros(len(y_split)), y_split):
        train_index = list(X.index.get_level_values('id').unique()[train_index])
        test_index = list(X.index.get_level_values('id').unique()[test_index])
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        if n_iter > 0:
            best_score = 0
            best_params = None
            for params in ParameterSampler(param_grid, n_iter, random_state=random_seed):
                scores = []
                if multitask == 'real_multitask':
                    skf_inner = KFold(n_splits=n_inner_splits, shuffle=True, random_state=random_seed)
                else:
                    skf_inner = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_seed)
                if multitask == 'multitask':
                    y_train_split = y_train[outcome]
                else:
                    y_train_split = y_train
                for train_index1, val_index in skf_inner.split(np.zeros(len(y_train_split)), y_train_split):
                    train_index1 = list(X_train.index.get_level_values('id').unique()[train_index1])
                    val_index = list(X_train.index.get_level_values('id').unique()[val_index])
                    X_train1, X_val = X_train.loc[train_index1], X_train.loc[val_index]
                    y_train1, y_val = y_train.loc[train_index1], y_train.loc[val_index]
                    if imputation_method is not None:
                        X_train1, X_val = data_imputation.impute_data(X_train1, X_val, imputation_method)
                    if model == 'lstm' or model == 'gru':
                        np.random.seed(random_seed)
                        tf.random.set_seed(random_seed)                        
                        len_time = int(len(X_train1) / len(X_train1.index.get_level_values('id').unique()))
                        len_columns = len(X_train1.columns)
                        n_cells = params['n_cells']
                        dropout_rate = params['dropout_rate']
                        recurrent_dropout_rate = params['recurrent_dropout_rate']
                        learning_rate = params['learning_rate']
                        epochs = params['epochs']
                        batch_size = params['batch_size']
                        tf.keras.backend.clear_session()
                        clf = KerasClassifier(rnn, model_type=model, len_time=len_time, len_columns=len_columns, 
                                              n_cells=n_cells, dropout_rate=dropout_rate,
                                              recurrent_dropout_rate=recurrent_dropout_rate,
                                              learning_rate=learning_rate)
                        pipe = Pipeline([
                                ('scaler', StandardScaler1()),
                                ('model', clf)
                                ])
                        pipe.fit(X_train1, y_train1, model__epochs=epochs, model__batch_size=batch_size,
                                 model__verbose=0)
                    else:
                        if scale:
                            pipe = Pipeline([
                                    ('scaler', StandardScaler()),
                                    ('model', initialize_model(model, multitask).set_params(random_state=random_seed, **params))
                                    ]) 
                        else:
                            pipe = initialize_model(model, multitask).set_params(random_state=random_seed, **params)

                        pipe.fit(X_train1, y_train1)
                    if multitask == 'real_multitask' or multitask == 'multitask':
                        y_val_pred = pipe.predict_proba(X_val)[:, 0]
                        scores.append(roc_auc_score(y_val[outcome], y_val_pred))
                    else:
                        y_val_pred = pipe.predict_proba(X_val)[:, 1]
                        scores.append(roc_auc_score(y_val, y_val_pred))
                score = np.mean(scores)
                if score > best_score:
                    best_score = score
                    best_params = params
            if model == 'lstm' or model == 'gru':
                np.random.seed(random_seed)
                tf.random.set_seed(random_seed)
                len_time = int(len(X_train) / len(X_train.index.get_level_values('id').unique()))
                len_columns = len(X_train.columns)
                n_cells = best_params['n_cells']
                dropout_rate = best_params['dropout_rate']
                recurrent_dropout_rate = params['recurrent_dropout_rate']
                learning_rate = best_params['learning_rate']
                epochs = best_params['epochs']
                batch_size = best_params['batch_size']
                clf = KerasClassifier(rnn, model_type=model, len_time=len_time, len_columns=len_columns, 
                                      n_cells=n_cells, dropout_rate=dropout_rate,
                                      recurrent_dropout_rate=recurrent_dropout_rate,
                                      learning_rate=learning_rate)
                pipe = Pipeline([
                        ('scaler', StandardScaler1()),
                        ('model', clf)
                        ])
                pipe.fit(X_train, y_train, model__epochs=epochs, model__batch_size=batch_size, 
                         model__verbose=0)
            else:
                if scale:
                    pipe = Pipeline([
                            ('scaler', StandardScaler()),
                            ('model', initialize_model(model, multitask).set_params(random_state=random_seed, **best_params))
                            ]) 
                else:
                    pipe = initialize_model(model, multitask).set_params(random_state=random_seed, **best_params)
                    
                if imputation_method is not None:
                    X_train, X_test = data_imputation.impute_data(X_train, X_test, imputation_method)
                pipe.fit(X_train, y_train)

        else:
            if imputation_method is not None:
                X_train, X_test = data_imputation.impute_data(X_train, X_test, imputation_method)
            if scale:
                pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', initialize_model(model, multitask).set_params(random_state=random_seed))
                        ]) 
            else:
                pipe = Pipeline([
                        ('model', initialize_model(model, multitask).set_params(random_state=random_seed))
                        ]) 
            pipe.fit(X_train, y_train)
        
        if multitask == 'multitask' or multitask  == 'real_multitask':
            y_pred_all.append(pipe.predict_proba(X_test))
        else:
            y_pred_all.append(pipe.predict_proba(X_test)[:, 1])
        y_test_all.append(np.array(y_test))
    
    results = {'y_test': y_test_all, 'y_pred': y_pred_all}
    if store:
        np.save(f'results/results_{population}_{outcome}_{model}_{random_seed}_{file_ending1}.npy', results)    
            
    return results

def compute_shap_values(outcome,
                        population,
                        data,
                        n_iter,
                        n_splits,
                        param_grid,
                        random_seed,
                        file_ending):
    X = data.drop('y', axis=1)
    y = data['y']
    model = 'catboost'
    best_score = 0
    best_params = None
    for params in ParameterSampler(param_grid, n_iter, random_state=random_seed):
        scores = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            clf = initialize_model(model).set_params(random_state=random_seed, **params)
            clf.fit(X_train, y_train)
            y_val_pred = clf.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, y_val_pred))
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_params = params

    clf = initialize_model(model).set_params(random_state=random_seed, **best_params)
    clf.fit(X, y)
        
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    np.save(f'shap_data/shap_values_{population}_{outcome}_{file_ending}2.npy', shap_values)
    
    return 0

def compute_lr_coefs(outcome,
                     population,
                     data,
                     n_iter,
                     n_outer_splits,
                     n_inner_splits,
                     param_grid,
                     random_seeds,
                     file_ending,
                     imputation_method,
                     store):
    X = data.drop('y', axis=1)
    y = data['y']
    model = 'logreg'
    best_score = 0
    best_params = None
    coefs = []
    for random_seed in random_seeds:
        skf_outer = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=random_seed)
        for train_index, test_index in skf_outer.split(np.zeros(len(y)), y):
            train_index = list(X.index.get_level_values('id').unique()[train_index])
            test_index = list(X.index.get_level_values('id').unique()[test_index])
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            best_score = 0
            best_params = None
            for params in ParameterSampler(param_grid, n_iter, random_state=random_seed):
                scores = []
                skf_inner = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_seed)
                for train_index1, val_index in skf_inner.split(np.zeros(len(y_train)), y_train):
                    train_index1 = list(X_train.index.get_level_values('id').unique()[train_index1])
                    val_index = list(X_train.index.get_level_values('id').unique()[val_index])
                    X_train1, X_val = X_train.loc[train_index1], X_train.loc[val_index]
                    y_train1, y_val = y_train.loc[train_index1], y_train.loc[val_index]
                    X_train1, X_val = data_imputation.impute_data(X_train1, X_val, imputation_method)
                    pipe = Pipeline([
                            ('scaler', StandardScaler()),
                            ('model', initialize_model(model).set_params(random_state=random_seed, **params))
                            ]) 

                    pipe.fit(X_train1, y_train1)
                    y_val_pred = pipe.predict_proba(X_val)[:, 1]
                    scores.append(roc_auc_score(y_val, y_val_pred))
                score = np.mean(scores)
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            pipe = Pipeline([
                            ('scaler', StandardScaler()),
                            ('model', initialize_model(model).set_params(random_state=random_seed, **best_params))
                            ]) 
            X_train, X_test = data_imputation.impute_data(X_train, X_test, imputation_method)
            pipe.fit(X_train, y_train)
            coefs.append(pipe['model'].coef_[0])
    if store:        
        np.save(f'lr_coefs/lr_coefs_{population}_{outcome}_{file_ending}.npy', coefs)
    
    return coefs