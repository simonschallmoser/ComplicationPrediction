import numpy as np
import pandas as pd
import pickle 

outcome_codes = [585, 362, 357, 443, 440, 430, 431, 432, 433, 434, 435, 437, 438, 410, 411, 412, 413, 414]

def fill_categorical_to_one_hot(df,
                                col_dict):
    """
    This function replaces categorical variables by one-hot encoded matrizes in a pandas df
    Input: Pandas dataframe, dictionary of column to categories which should be one-hotted.
    Output: One-hot-encoded dataframe.
    """
    index = df.index
    n_cat_cols = np.sum([len(df[col].unique()) for col in df.columns if col in col_dict])
    #print('Having {} columns to fill'.format(n_cat_cols))
    fill_df = np.zeros((len(df),n_cat_cols), dtype=np.float32)
    fill_non_cat_df = []
    column_cat_names = []
    column_non_cat_names = []
    column_id = 0
    for col in df.columns:
        if col in col_dict:
            #print('One-hotting {}'.format(col))
            class_to_idx = dict(zip(col_dict[col], np.arange(len(col_dict[col]))))
            indizes = df[col].map(class_to_idx).astype(np.int32)
            fill_df[np.arange(df.shape[0]), column_id + indizes] = 1.
            column_cat_names.extend([str(col) + "=" + str(x) for x in col_dict[col]])
            column_id += len(col_dict[col])
        else:
            #print('Keeping {} as is'.format(col))
            fill_non_cat_df.append(df[col])
            column_non_cat_names.append(col)
    #print('Merging everything together')
    one_hotted_df = pd.DataFrame(data=fill_df, columns=column_cat_names)
    for col_name, col in zip(column_non_cat_names, fill_non_cat_df):
        one_hotted_df[col_name] = col.values
    one_hotted_df.index = index
    return one_hotted_df

def outcome_code(outcome):
    """
    Transform outcome into outcome ICD-9 codes
    """
    if outcome == 'renal':
        outcome_icd9_codes = [250.4, 585]
    elif outcome == 'eyes':
        outcome_icd9_codes = [250.5, 362.0]
    elif outcome == 'nerves':
        outcome_icd9_codes = [250.6, 357.2]
    elif outcome == 'pvd':
        outcome_icd9_codes = [250.7, 443.9, 440]
    elif outcome == 'cevd':
        outcome_icd9_codes = [430, 431, 432, 433, 434, 435, 437, 438]
    elif outcome == 'cavd':
        outcome_icd9_codes = [410, 411, 412, 413, 414]
    else:
        raise ValueError('Not defined outcome.')
        
    return outcome_icd9_codes

def compute_ids_prediabetes_onset(df_lab,
                                  df_icd,
                                  df_demo,
                                  ids_diabetes,
                                  outcome,
                                  baseline=2008):
    """
    Computes the IDs of patients who are diagnosed with prediabetes
    according to their HbA1c (%) values or ICD-9 code.
    Prediabetes: 5.7% <= HbA1c < 6.5% or ICD-9 code of 790.2
    Input: Lab data and ICD-9 data.
    Output: Dictionary with IDs as keys and year of diagnosis as value.
    """
    df_lab_prediabetes = df_lab[(df_lab['test'] == 'hba1c(%)') & (df_lab['test_outcome'] >= 5.7) 
                             & (df_lab['test_outcome'] < 6.5)]
    df_lab_prediabetes = df_lab_prediabetes.reset_index().groupby('id').min().year

    df_icd_prediabetes = df_icd[df_icd.icd9.apply(lambda x: '790.2' in x)].reset_index().groupby('id').min().year
    df_prediabetes = df_icd_prediabetes.to_frame().rename({'year': 'year_icd'}, axis=1).join(df_lab_prediabetes, 
                                                                                              how='outer')
    ids_prediabetes = dict(df_prediabetes.min(axis=1))
    demo_ids = list(df_demo.index.get_level_values('id').unique())
    ids_prediabetes = {k:ids_prediabetes[k] for k in ids_prediabetes.keys() if k in demo_ids and
                       ids_prediabetes[k] <= baseline}
    ids_prediabetes = {k:ids_prediabetes[k] for k in ids_prediabetes.keys() if k not in ids_diabetes.keys()}
    if outcome == 'renal':
        max_years_icd = df_icd.reset_index().groupby('id').year.max().to_frame()
        df_lab_sub = df_lab[(df_lab['test'] == 'creatinineserum') | (df_lab['test'] == 'albumin/creatineratio-u')]
        max_years_lab = df_lab_sub.reset_index().groupby('id').year.max().to_frame().rename({'year': 'year_lab'}, axis=1)
        max_years = max_years_icd.join(max_years_lab, how='outer')
        max_years = dict(df_icd.reset_index().groupby('id').year.max())
    else:
        max_years = dict(df_icd.reset_index().groupby('id').year.max())
    ids_prediabetes = {k:ids_prediabetes[k] for k in ids_prediabetes.keys() if k in max_years.keys() and
                       max_years[k] == 2013}
    
    return ids_prediabetes

def compute_ids_diabetes_onset(df_lab,
                               df_icd,
                               df_demo,
                               df_med,
                               antidiabetes_medication,
                               outcome,
                               baseline=2008):
    """
    Computes the IDs of patients who are diagnosed with diabetes
    according to their HbA1c (%) values, ICD-9 codes or prescribed antidiabetes medication.
    Diabetes: Two measurements of HbA1c >= 6.5%, ICD-9 code of 250.xx or prescribed antidiabetes medication. 
    Input: Lab data, ICD-9 data, medication data and list of antidiabetes medications.
    Output: Dictionary with IDs as keys and year of diagnosis as value.
    """
    df_lab_diabetes = df_lab[(df_lab['test'] == 'hba1c(%)') & (df_lab['test_outcome'] >= 6.5)]
    df_lab_diabetes = df_lab_diabetes.reset_index().set_index('id')
    df_lab_diabetes = df_lab_diabetes[df_lab_diabetes.index.duplicated(keep=False)]
    df_lab_diabetes = df_lab_diabetes.groupby('id').year.min()
    
    df_icd_diabetes = df_icd[df_icd.icd9_prefix.isin([250])].reset_index().groupby('id').year.min()
    
    df_med_diabetes = df_med[df_med.med.isin(antidiabetes_medication)].reset_index().groupby('id').year.min()
    
    df_diabetes = df_icd_diabetes.to_frame().rename({'year': 'year_icd'}, axis=1).join(df_lab_diabetes, 
                                                                                              how='outer')
    df_diabetes = df_diabetes.rename({'year': 'year_lab'}, axis=1).join(df_med_diabetes, how='outer').min(axis=1)
    ids_diabetes = dict(df_diabetes)
    demo_ids = list(df_demo.index.get_level_values('id').unique())
    ids_diabetes = {k:ids_diabetes[k] for k in ids_diabetes.keys() if k in demo_ids and 
                          ids_diabetes[k] <= baseline}
    if outcome == 'renal':
        max_years_icd = df_icd.reset_index().groupby('id').year.max().to_frame()
        df_lab_sub = df_lab[(df_lab['test'] == 'creatinineserum') | (df_lab['test'] == 'albumin/creatineratio-u')]
        max_years_lab = df_lab_sub.reset_index().groupby('id').year.max().to_frame().rename({'year': 'year_lab'}, axis=1)
        max_years = max_years_icd.join(max_years_lab, how='outer')
        max_years = dict(df_icd.reset_index().groupby('id').year.max())
    else:
        max_years = dict(df_icd.reset_index().groupby('id').year.max())
    ids_diabetes = {k:ids_diabetes[k] for k in ids_diabetes.keys() if max_years[k] == 2013}
    
    return ids_diabetes

def estimated_gfr(df_lab,
                  df_demo,
                  egfr_method='ckd-epi'):
    """
    Estimates the glomerular filtration rate based on the formula from 
    Levey (2009) if method is ckd-epi
    Levey (1999) if method is mdrd.
    """
    lab_scr = df_lab[df_lab['test'] == 'creatinineserum']
    df = lab_scr.join(df_demo[['birthyear', 'sex']])
    df['sex'] = df.groupby(axis=0, level='id').sex.ffill().bfill()
    df['birthyear'] = df.groupby(axis=0, level='id').birthyear.ffill().bfill()
    df['age'] = df.index.get_level_values('year') - df['birthyear']
    if egfr_method == 'ckd-epi':
        df['kappa'] = [0.7 if i == 1 else 0.9 for i in df['sex']]
        df['alpha'] = [-0.329 if i == 1 else -0.411 for i in df['sex']]
        df['1'] = 1
        df['scr/kappa'] = df['test_outcome'] / df['kappa']
        df['min'] = df[['scr/kappa', '1']].min(axis=1)
        df['max'] = df[['scr/kappa', '1']].max(axis=1)
        df['egfr'] = 141 * df['min'] ** df['alpha'] \
                         * df['max'] ** -1.209 \
                         * 0.993 ** df['age'] * (1 + df['sex'] * (1.018 - 1))
    elif egfr_method == 'mdrd':
        df['egfr'] = 186 * df['test_outcome'] ** (-1.154) * df['age'] ** (-0.203) * \
                         (1 + df['sex'] * (0.742 - 1))
    else:
        raise ValueError('Unknown method for estimating glomerular filtration rate. Choose either ckd-epi or mdrd.')
    
    return df['egfr']

def compute_ids_outcome_onset(df_icd,
                              df_lab,
                              ids_pre_diabetes,
                              outcomes,
                              pred_year,
                              baseline=2008,
                              estimate_gfr=False,
                              df_demo=None,
                              alb_scr_level=30,
                              egfr_level=60,
                              multilabel=False,
                              include_lab_nephropathy=True,
                              egfr_method='ckd-epi'):
    """
    Computes the IDs of patients who developed a complication corresponding to recorded ICD-9 codes 
    (or certain biomarkers for nephropathy, namely UACR and optionally the GFR estimated from SCr measurements)
    Input: Lab data, ICD-9 data, IDs of diabetes patients, ICD-9 codes for specific complication, and forecast horizon.
    Output: Dictionaries with IDs as keys and year of diagnosis as value for (pre-)diabetes patients and patients with complication.
    """
    codes = list(df_icd.icd9.unique())
    all_outcomes = [[i for i in codes if str(j) in i] for j in outcomes]
    all_outcomes = [i for sublist in all_outcomes for i in sublist]
    icd_outcome = df_icd[df_icd.icd9.apply(lambda row: row in all_outcomes)]
    
    # If outcome is nephropathy, include lab values of eGFR and albumin/creatinine in urine as disease defining markers
    if 250.4 in outcomes and include_lab_nephropathy:
        icd_outcome = icd_outcome.reset_index().set_index('id').year.to_frame()
        icd_outcome = icd_outcome.rename({'year': 'year_icd'}, axis=1)

        # Define outcome by increased ratio of albumin and creatinine in urine - at least two 
        # increased tests have to be present
        alb_scr = df_lab[df_lab['test'] == 'albumin/creatineratio-u']
        alb_scr_incr = alb_scr[alb_scr['test_outcome'] >= alb_scr_level]
        alb_scr_incr = alb_scr_incr.reset_index().set_index('id')
        alb_scr_incr_twice = alb_scr_incr[alb_scr_incr.index.duplicated(keep=False)]
        alb_scr_incr_twice = alb_scr_incr_twice.reset_index().set_index('id')['year'].to_frame()
        alb_scr_incr_twice = alb_scr_incr_twice.rename({'year': 'year1'}, axis=1)

        # Define outcome by decreased estimated glomerular filtration rate
        #egfr = df_lab[df_lab['test'] == 'gfr(estimated)']
        #egfr_decr = egfr[egfr['test_outcome'] < egfr_level]
        #egfr_decr = egfr_decr.reset_index().set_index('id')['year'].to_frame()
        
        # Define outcome by estimated glomerular filtration rate (egfr) using serum creatinine measurements
        if estimate_gfr:
            egfr = estimated_gfr(df_lab, df_demo, egfr_method)
            egfr_decr = egfr[egfr < egfr_level]
            egfr_decr = egfr_decr.reset_index().set_index('id')['year'].to_frame().rename({'year': 'year2'}, axis=1)

        # Join dataframes
        #df_joined = alb_scr_incr_twice.join(egfr_decr, how='outer')
        if estimate_gfr:
            #df_joined = egfr_decr.join(df_joined, how='outer')
            df_joined = egfr_decr.join(alb_scr_incr_twice, how='outer')
            df_joined = icd_outcome.join(df_joined, how='outer')
        else:
            df_joined = icd_outcome.join(alb_scr_incr_twice, how='outer')

        # Get minimal year of outcome definition
        df_joined = df_joined.groupby(axis=0, level='id').min().min(axis=1)

        #Transform to dictionary
        outcomes_dict = dict(df_joined)
        
    else: 
        outcomes_dict = dict(icd_outcome.reset_index().groupby('id').year.min())
    
    if multilabel:
        return outcomes_dict
        
    # Identify those IDs with outcome diagnosis after 2008 and only those who have been
    # identified as diabetic beforehand
    ids_outcome_final = {k:outcomes_dict[k] for k in outcomes_dict.keys() if 
                               (k in ids_pre_diabetes.keys()) and
                               (outcomes_dict[k] > baseline) and 
                               (outcomes_dict[k] <= baseline + pred_year)}
    
    # Only consider those diabetes patients who have not been diagnosed with the outcome
    # before 2009 
    ids_pre_diabetes_final = {k:ids_pre_diabetes[k] for k in ids_pre_diabetes.keys() if k not in outcomes_dict.keys() or 
                              outcomes_dict[k] > baseline}  
    
    return ids_pre_diabetes_final, ids_outcome_final