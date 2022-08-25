import numpy as np
import pandas as pd
import miceforest as mf

def impute_demo(data, ffill_demo=True):
    df = data.copy()
    df['birthyear'] = df.birthyear.unstack(0).ffill().bfill().stack().swaplevel(0,1).groupby(['id','year']).first()
    df['age'] = df.index.get_level_values('year').to_series().apply(lambda row: int(str(row)[:4])).values - df['birthyear']
    df['sex'] = df.sex.unstack(0).ffill().bfill().stack().swaplevel(0,1).groupby(['id','year']).first()
    if ffill_demo:
        df['systolic_bp'] = df.systolic_bp.unstack(0).ffill().stack().swaplevel(0,1).groupby(['id','year']).first()
        df['diastolic_bp'] = df.diastolic_bp.unstack(0).ffill().stack().swaplevel(0,1).groupby(['id','year']).first()
        df['bmi'] = df.bmi.unstack(0).ffill().stack().swaplevel(0,1).groupby(['id','year']).first()
    else:
        df['systolic_bp'] = df.systolic_bp.unstack(0).stack().swaplevel(0,1).groupby(['id','year']).first()
        df['diastolic_bp'] = df.diastolic_bp.unstack(0).stack().swaplevel(0,1).groupby(['id','year']).first()
        df['bmi'] = df.bmi.unstack(0).stack().swaplevel(0,1).groupby(['id','year']).first()
    
    cols_to_drop = ['birthyear']
    
    return df.drop(cols_to_drop, axis=1)

def impute_data(X_train, X_test, method, mice_iterations=3):
    
    train = X_train.copy()
    test = X_test.copy()

    cols_to_impute_demo = ['systolic_bp', 'diastolic_bp', 'bmi']
    cols_to_impute_demo = [i for j in cols_to_impute_demo for i in train.columns if j in i]
    cols_to_impute_lab = [i for i in train.columns if 'test' in i]
    cols_to_impute_icd = [i for i in train.columns if 'icd9' in i]
    cols_to_impute_med = [i for i in train.columns if 'med' in i]
    
    if method == 'mean':
        means_demo = train[cols_to_impute_demo].mean()
        means_lab = train[cols_to_impute_lab].mean()
        for col_demo in cols_to_impute_demo:
            train[col_demo] = train[col_demo].replace(np.nan, means_demo.loc[col_demo])
            test[col_demo] = test[col_demo].replace(np.nan, means_demo.loc[col_demo])
        for col_lab in cols_to_impute_lab:
            train[col_lab] = train[col_lab].replace(np.nan, means_lab.loc[col_lab])
            test[col_lab] = test[col_lab].replace(np.nan, means_lab.loc[col_lab])
    elif method == 'median':
        medians_demo = train[cols_to_impute_demo].median()
        medians_lab = train[cols_to_impute_lab].median()
        for col_demo in cols_to_impute_demo:
            train[col_demo] = train[col_demo].replace(np.nan, medians_demo.loc[col_demo])
            test[col_demo] = test[col_demo].replace(np.nan, medians_demo.loc[col_demo])
        for col_lab in cols_to_impute_lab:
            train[col_lab] = train[col_lab].replace(np.nan, medians_lab.loc[col_lab])
            test[col_lab] = test[col_lab].replace(np.nan, medians_lab.loc[col_lab])
    elif method == 'mice':
        kernel = mf.ImputationKernel(
                      train,
                      save_all_iterations=False,
                      random_state=0
                      )
        kernel.mice(mice_iterations)
        train = kernel.complete_data(dataset=4, inplace=False)
        test = kernel.impute_new_data(new_data=test)
        test = test.complete_data(dataset=4, inplace=False)
    else:
        raise NameError('Type of imputation method not known. \
                        Choose either mean, median or mice.')
    
    train[cols_to_impute_icd] = train[cols_to_impute_icd].replace(np.nan, 0.0)
    test[cols_to_impute_icd] = test[cols_to_impute_icd].replace(np.nan, 0.0)
    train[cols_to_impute_med] = train[cols_to_impute_med].replace(np.nan, 0.0)
    test[cols_to_impute_med] = test[cols_to_impute_med].replace(np.nan, 0.0)
    
    return train, test
