import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, precision_recall_curve

import helper
from categories_dicts import bucketize_icds, test_name_to_bucket, icd9bucket_to_name, med_to_categories, med_to_class
from data_imputation import impute_demo

def preprocessing_icd9_df(df):
    """
    One-hot encodes ICD9 data (prefixes and buckets),
    takes only those codes which appear at least 200 times,
    adds columns with ICD9 codes of previous year.
    Input: ICD9 data.
    Output: Preprocessed ICD9 data.
    """
    df_icd9 = df.copy()
    value_counts = df_icd9.icd9_prefix.value_counts()
    to_remove = value_counts[value_counts < 50].index
    df_icd9.icd9_prefix.replace(to_remove, np.nan, inplace=True)
    df_icd9.dropna(inplace=True)
    df_icd9['icd9_prefix'] = df_icd9['icd9_prefix'].astype(int)
    df_icd9 = helper.fill_categorical_to_one_hot(df_icd9.drop('icd9', axis=1), 
                                                 {'icd9_bucket': df_icd9.icd9_bucket.unique(), 
                                                  'icd9_prefix': df_icd9.icd9_prefix.unique()})
    df_icd9 = df_icd9.groupby(df_icd9.index).sum()
    df_icd9.index = pd.MultiIndex.from_tuples(df_icd9.index)
    df_icd9.index.names = ['id', 'year']
    
    return df_icd9.astype(bool).astype(int)

def add_lab_trajectory(df_lab_sub,
                       baseline,
                       only_trend=False):
    """
    Adds seven representations of the trajectory for each biomarker:
    min, max, mean, median, std, trend (last-first) and trend1 (max-min).
    """
    df_lab_sub1 = df_lab_sub.loc[pd.IndexSlice[:, :baseline], :].copy()
    df_lab_max = df_lab_sub1.groupby(axis=0, level='id').max()
    df_lab_min = df_lab_sub1.groupby(axis=0, level='id').min()
    df_lab_mean = df_lab_sub1.groupby(axis=0, level='id').mean()
    df_lab_std = df_lab_sub1.groupby(axis=0, level='id').std()
    df_lab_last = df_lab_sub1.groupby(axis=0, level='id').last()
    df_lab_first = df_lab_sub1.groupby(axis=0, level='id').first()
    df_lab_base = df_lab_sub1.loc[pd.IndexSlice[:, baseline], :].groupby(axis=0, level='id').mean()
    df_lab_base_minus_two = df_lab_sub1.loc[pd.IndexSlice[:, baseline-2], :].groupby(axis=0, level='id').mean()
    df_lab_n_measurements = df_lab_sub1.groupby(axis=0, level='id').count()
    df_lab_trend = df_lab_last.sub(df_lab_first)
    df_lab_trend1 = df_lab_max.sub(df_lab_min)
    df_lab_trend2 = df_lab_base.sub(df_lab_base_minus_two)
    
    df_lab_max = df_lab_max.rename({i:i+'_max' for i in df_lab_sub1.columns}, axis=1)
    df_lab_min = df_lab_min.rename({i:i+'_min' for i in df_lab_sub1.columns}, axis=1)
    df_lab_mean = df_lab_mean.rename({i:i+'_mean' for i in df_lab_sub1.columns}, axis=1)
    df_lab_std = df_lab_std.rename({i:i+'_std' for i in df_lab_sub1.columns}, axis=1)
    df_lab_n_measurements = df_lab_n_measurements.rename({i:i+'_n_measurements' for i in df_lab_sub1.columns}, axis=1)
    df_lab_trend = df_lab_trend.rename({i:i+'_trend' for i in df_lab_sub1.columns}, axis=1)
    df_lab_trend1 = df_lab_trend1.rename({i:i+'_trend1' for i in df_lab_sub1.columns}, axis=1)
    df_lab_trend2 = df_lab_trend2.rename({i:i+'_trend2' for i in df_lab_sub1.columns}, axis=1)
    
    if only_trend:
        df_lab_trend.index = pd.MultiIndex.from_tuples(((i, baseline) for i in df_lab_trend.index), names=['id', 'year'])
        return df_lab_trend
    
    df_lab_all = pd.concat([df_lab_trend2, df_lab_trend1, df_lab_trend, df_lab_mean, df_lab_n_measurements, 
                            df_lab_std, df_lab_min, df_lab_max], axis=1)
    df_lab_all.index = pd.MultiIndex.from_tuples(((i, baseline) for i in df_lab_all.index), names=['id', 'year'])
    
    return df_lab_all

def add_demo_trajectory(df_demo_sub,
                        baseline,
                        only_trend=False):
    """
    Adds seven representations of the trajectory for the demographic predictors (BMI, systolic and diastolic blood pressure):
    min, max, mean, median, std, trend (last-first) and trend1 (max-min).
    """
    columns_trajectory = ['bmi', 'systolic_bp', 'diastolic_bp']
    df_demo_sub1 = df_demo_sub.loc[pd.IndexSlice[:, :baseline], :].copy()[columns_trajectory]
    df_demo_max = df_demo_sub1.groupby(axis=0, level='id').max()
    df_demo_min = df_demo_sub1.groupby(axis=0, level='id').min()
    df_demo_mean = df_demo_sub1.groupby(axis=0, level='id').mean()
    df_demo_std = df_demo_sub1.groupby(axis=0, level='id').std()
    df_demo_last = df_demo_sub1.groupby(axis=0, level='id').last()
    df_demo_first = df_demo_sub1.groupby(axis=0, level='id').first()
    df_demo_base = df_demo_sub1.loc[pd.IndexSlice[:, baseline], :].groupby(axis=0, level='id').mean()
    df_demo_base_minus_two = df_demo_sub1.loc[pd.IndexSlice[:, baseline-2], :].groupby(axis=0, level='id').mean()
    df_demo_n_measurements = df_demo_sub1.groupby(axis=0, level='id').count()
    df_demo_trend = df_demo_last.sub(df_demo_first)
    df_demo_trend1 = df_demo_max.sub(df_demo_min)
    df_demo_trend2 = df_demo_base.sub(df_demo_base_minus_two)
    
    df_demo_max = df_demo_max.rename({i:i+'_max' for i in df_demo_sub1.columns}, axis=1)
    df_demo_min = df_demo_min.rename({i:i+'_min' for i in df_demo_sub1.columns}, axis=1)
    df_demo_mean = df_demo_mean.rename({i:i+'_mean' for i in df_demo_sub1.columns}, axis=1)
    df_demo_std = df_demo_std.rename({i:i+'_std' for i in df_demo_sub1.columns}, axis=1)
    df_demo_n_measurements = df_demo_n_measurements.rename({i:i+'_n_measurements' for i in df_demo_sub1.columns}, axis=1)
    df_demo_trend = df_demo_trend.rename({i:i+'_trend' for i in df_demo_sub1.columns}, axis=1)
    df_demo_trend1 = df_demo_trend1.rename({i:i+'_trend1' for i in df_demo_sub1.columns}, axis=1)
    df_demo_trend2 = df_demo_trend2.rename({i:i+'_trend2' for i in df_demo_sub1.columns}, axis=1)
    
    if only_trend:
        df_demo_trend.index = pd.MultiIndex.from_tuples(((i, baseline) for i in df_demo_trend.index), names=['id', 'year'])
        return df_demo_trend
    
    df_demo_all = pd.concat([df_demo_trend2, df_demo_trend1, df_demo_trend, df_demo_mean, df_demo_n_measurements,
                             df_demo_std, df_demo_min, df_demo_max], axis=1)
    df_demo_all.index = pd.MultiIndex.from_tuples(((i, baseline) for i in df_demo_all.index), names=['id', 'year'])
    
    return df_demo_all

def preprocessing_lab_df(df, 
                         baseline,
                         add_trajectory=False,
                         only_trend=False, 
                         value='mean',
                         date_type='year',
                         ffill=True,
                         n_steps=5):
    """
    One hot encodes the tests that have been done
    and multiplies with the test outcome.
    Done in a loop of 5 steps in order to not overload RAM.
    Keeps only tests which were performed at least 500 times.
    Input: Lab data.
    Output: Preprocessed lab data.
    """
    df_lab = df.copy()
    value_counts = df_lab.test.value_counts()
    to_remove = value_counts[value_counts < 5000].index
    df_lab.replace(to_remove, np.nan, inplace=True)
    df_lab.dropna(inplace=True)
    ids = list(df_lab.index.get_level_values('id').unique())
    length = int(len(ids)/n_steps)
    df_lab_total = []
    df_lab_trajectory = []
    for i in range(n_steps):
        #print('Preprocess chunk', i)
        if i == n_steps - 1:
            df_lab_sub = df_lab.loc[ids[length*i:]]
        else:
            df_lab_sub = df_lab.loc[ids[length*i:length*(i+1)]]
        df_lab_sub = helper.fill_categorical_to_one_hot(df_lab_sub, {'test': df_lab_sub.test.unique()})
        df_outcome = df_lab_sub.test_outcome.copy()
        df_lab_sub.drop('test_outcome', axis=1, inplace=True)
        df_lab_sub = df_lab_sub.mul(df_outcome, axis=0)
        df_lab_sub = df_lab_sub.replace(0.0, np.nan)
        if add_trajectory:
            df_lab_trajectory.append(add_lab_trajectory(df_lab_sub, baseline, only_trend))
        if value == 'mean':
            df_lab_sub = df_lab_sub.groupby(df_lab_sub.index).mean()
        elif value == 'last':
            df_lab_sub = df_lab_sub.groupby(df_lab_sub.index).last()
        else:
            raise ValueError('Unknown value type. Choose either mean or last.')
        df_lab_sub.index = pd.MultiIndex.from_tuples(df_lab_sub.index)
        df_lab_sub.index.names = ['id', 'year']
        df_lab_total.append(df_lab_sub)

    df_lab_total = pd.concat(df_lab_total)
    if add_trajectory:
        df_lab_trajectory = pd.concat(df_lab_trajectory)
        df_lab_total = df_lab_total.join(df_lab_trajectory, how='outer')
    if date_type == 'year':
        ids_baseline = df_lab_total.loc[pd.IndexSlice[:, baseline], :].index.get_level_values('id').unique()
        ids_baseline_missing = [i for i in ids if i not in ids_baseline]
        if len(ids_baseline_missing) != 0:
            tuples_missing = [(i, baseline) for i in ids_baseline_missing]
            df_conc = pd.DataFrame(data=None, index=pd.MultiIndex.from_tuples(tuples_missing))
    elif date_type == 'date':
        ids_baseline = df_lab_total.loc[pd.IndexSlice[:, int(str(baseline)+'0101'):int(str(baseline)+'1231')],
                                        :].index.get_level_values('id').unique()
        ids_baseline_missing = [i for i in ids if i not in ids_baseline]
        if len(ids_baseline_missing) != 0:
            tuples_missing = [(i, int(str(baseline)+'1231')) for i in ids_baseline_missing]
            df_conc = pd.DataFrame(data=None, index=pd.MultiIndex.from_tuples(tuples_missing))
    else:
        raise ValueError('Unknown date type. Choose eiter year or date.')
    if len(ids_baseline_missing) != 0:
        df_lab_total = pd.concat([df_lab_total, df_conc]).sort_index()
    if ffill:
        df_lab_total = df_lab_total.groupby(axis=0, level='id').ffill()

    return df_lab_total

def preprocessing_med_df(df):
    """
    Preprocessed medication data, one-hot encodes medications.
    Input: Medication data.
    Output: Preprocessed medication data.
    """
    df_med = df.copy()
    df_med = df_med.reset_index()
    df_med = df_med[df_med.med.isin(list(med_to_categories.keys()))]
    df_med['med_id'] = df_med.med.apply(lambda x: med_to_categories[x])
    df_med = df_med[df_med.med_id.isin(list(med_to_class.keys()))]
    df_med['med_class'] = df_med.med_id.apply(lambda x: med_to_class[x][0])
    df_med.drop(['med', 'med_id'], inplace=True, axis=1)
    
    df_med.reset_index(inplace=True, drop=True)
    expanded_categories = np.zeros(shape=(len(df_med),len(df_med.med_class.unique())), dtype=np.int32)
    for i, row in df_med.iterrows():
            expanded_categories[i][row['med_class']] += 1
    
    df_ext = pd.DataFrame(data = expanded_categories)
    df_ext.columns = ["(med_class = " + str(x) + ")" for x in range(len(df_med.med_class.unique()))]
    df_med = pd.concat([df_med, df_ext], axis=1)
    df_med.drop('med_class', inplace=True, axis=1)
    df_med.set_index(['id', 'year'])
    df_med = df_med.groupby(['id', 'year']).sum().astype(bool).astype(int)
    
    return df_med

def remove_outliers(data,
                    lower,
                    upper):
    """
    Remove data outside of lower, upper percentile ranges.
    Input: Dataframe, lower and upper bounds.
    Output: Dataframe with outliers replaced as nan."""
    cols = [i for i in data.columns if 'test' in i]
    if 'bmi' in data.columns:
        cols = np.append(cols, ['bmi'])
    if 'systolic_bp' in data.columns:
        cols = np.append(cols, 'systolic_bp')
    if 'diastolic_bp' in data.columns:
        cols = np.append(cols, 'diastolic_bp')
    df = data.copy()
    for col in cols:
        df[col].mask(~df[col].between(df[col].quantile(lower), df[col].quantile(upper)), inplace=True)
        
    return df

def impute_icd9_med_data(df):
    """
    Replace missing values for ICD-9 and medication data with 0.
    """
    icd_cols = [i for i in df.columns if 'icd' in i]
    med_cols = [i for i in df.columns if 'med' in i]
    other_cols = [i for i in df.columns if i not in icd_cols and i not in med_cols]
    df = df[other_cols].join(df[np.append(icd_cols, med_cols)].replace(np.nan, 0))
    
    return df

def smoking(df, 
           icd):
    """
    Adds predictor smoking (active or former).
    Input: Final dataframe, ICD9 data.
    Output: Final dataframe with added predictor smoking.
    """
    ids_smoking = list(icd[(icd['icd9'] == '305.1') | 
                           (icd['icd9'] == 'V15.82')].loc[pd.IndexSlice[:, :2008],
                                                          :].index.get_level_values('id').unique())
    df_ids = list(df.index.get_level_values('id'))
    df['smoking'] = [1 if i in ids_smoking else 0 for i in df_ids]
    
    return df

def preprocess_data(df_lab, 
                    df_demo,
                    ids_pre_diabetes_onset, 
                    df_icd=None, 
                    df_med=None, 
                    baseline=2008,
                    add_trajectory=False,
                    only_trend=False,
                    lab_value_method='mean', 
                    date_type='year', 
                    ffill_lab=True,
                    ffill_demo=True, 
                    ffill_med=0):
    """
    Preprocesses data.
    Input: Lab data, demo data, IDs of pre-/diabetes patients; 
    optional: ICD-9 data and medication data.
    Output: Preprocessed data for all timestamps excluding labels.
    """
    print('Preprocess data')
    # Select subsets of demo and lab data
    demo_subset = df_demo.loc[list(ids_pre_diabetes_onset.keys())].copy()
    if add_trajectory:
        demo_trajectory = add_demo_trajectory(demo_subset, baseline, only_trend)
        demo_subset_mean = demo_subset.groupby(axis=0, level=['id', 'year']).mean()
        demo_subset = demo_subset_mean.join(demo_trajectory, how='outer')
    else:
        demo_subset = demo_subset.groupby(axis=0, level=['id', 'year']).mean()
    lab_subset = df_lab.loc[list(ids_pre_diabetes_onset.keys())].copy()
    
    # Preprocess subset of lab data
    lab_subset_preprocessed = preprocessing_lab_df(lab_subset, baseline=baseline,
                                                   add_trajectory=add_trajectory,
                                                   only_trend=only_trend,
                                                   value=lab_value_method, 
                                                   date_type=date_type,
                                                   ffill=ffill_lab)
    
    # Join demo and lab data  
    df = lab_subset_preprocessed.join(demo_subset, how='outer')
    
    if isinstance(df_icd, pd.DataFrame):
        # Select subset of ICD-9 data
        icd_subset = df_icd.loc[list(ids_pre_diabetes_onset.keys())].copy()
        # Preprocess subset of ICD-9 data
        icd_subset_prep = preprocessing_icd9_df(icd_subset)
        df = df.join(icd_subset_prep, how='outer')
    
    if isinstance(df_med, pd.DataFrame):
        # Select subset of medication data
        med_ids = df_med.index.get_level_values('id').unique()
        med_ids = [i for i in ids_pre_diabetes_onset.keys() if i in med_ids]
        med_subset = df_med.loc[med_ids]
        #Preprocess subset of medication data
        med_subset_prep = preprocessing_med_df(med_subset)
        if ffill_med > 0:
            med_subset_prep = med_subset_prep.groupby(axis=0, level='id').ffill(limit=ffill_med)
        df = df.join(med_subset_prep, how='outer')
    
    # Add missing indices for baseline
    if date_type == 'year':
        ids_baseline = list(df.loc[pd.IndexSlice[:, baseline], :].index.get_level_values('id').unique())
        ids_baseline_missing = [i for i in ids_pre_diabetes_onset.keys() if i not in ids_baseline]
        if len(ids_baseline_missing) != 0:
            tuples_missing = [(i, baseline) for i in ids_baseline_missing]
            df_conc = pd.DataFrame(data=None, index=pd.MultiIndex.from_tuples(tuples_missing))
            df = pd.concat([df, df_conc]).sort_index()
    elif date_type == 'date':
        ids_baseline = df.loc[pd.IndexSlice[:, int(str(baseline)+'0101'):int(str(baseline)+'1231')],
                                        :].index.get_level_values('id').unique()
        ids_baseline_missing = [i for i in ids_pre_diabetes_onset.keys() if i not in ids_baseline]
        if len(ids_baseline_missing) != 0:
            tuples_missing = [(i, int(str(baseline)+'1231')) for i in ids_baseline_missing]
            df_conc = pd.DataFrame(data=None, index=pd.MultiIndex.from_tuples(tuples_missing))
            df = pd.concat([df, df_conc]).sort_index()
    
    # Impute demo data: Change from birthyear to age, ffill blood pressure and BMI
    df = impute_demo(df, ffill_demo)
    
    return df

def final_preprocessing_step(data,
                             ids_pre_diabetes_onset,
                             ids_outcome_onset,
                             pred_year,
                             df_icd,
                             baseline=2008, 
                             predictors=None,
                             add_trajectory=False,
                             add_smoking=False,
                             time_since_diagnosis=False,
                             n_icd9_codes=50,
                             n_tests=60,
                             icd9_buckets=False, 
                             drop_icd9_250=False,
                             ffill_icd=True,
                             outlier_removal=True, 
                             lower=0.001,
                             upper=0.999):
    """
    Final preprocessing step selects data from baseline year and adds label depending on forecast horizon 
    (pred_year).
    Input: Preprocessed data, IDs for pre-/diabetes and outcome patients and forecast horizon (pred_year).
    Output: Final preprocessed data including labels.
    """
    print('Final preprocessing step.')
    
    if drop_icd9_250:
        n_icd9_codes += 1
        
    if ffill_icd:
        icd_cols = [i for i in data.columns if 'icd' in i]
        data[icd_cols] = data[icd_cols].replace(0, np.nan).groupby(axis=0, level='id').ffill().replace(np.nan, 0)
    
    # Only consider data at baseline
    df_final = data.loc[pd.IndexSlice[:, baseline], :].copy()
    
    # If specific columns are wanted
    if predictors is not None:
        if add_trajectory:
            predictors = [i for j in predictors for i in df_final.columns if j in i]
        df_final = df_final[predictors]
        
        # Only consider those patients with (pre) diabetes who have not been diagnosed with the outcome
        # before baseline
        df_final = df_final.loc[list(ids_pre_diabetes_onset.keys())]
        
    else:
        # Drop those ICD-9 codes which are associated to an outcome
        columns = list(df_final.columns)
        icd9_columns = [i for i in columns if 'icd9_prefix' in i]
        icds_to_drop = [j for i in helper.outcome_codes for j in icd9_columns if str(i) in j]
        df_final = df_final.drop(icds_to_drop, axis=1)
        
        # Only consider most common ICD-9 codes and lab tests
        columns = list(df_final.columns)
        test_columns = [i for i in columns if 'test' in i and '_' not in i]
        icd9_columns = [i for i in columns if 'icd9_prefix' in i]

        tests_to_drop = list(df_final[test_columns].isna().sum().sort_values(ascending=True)[n_tests:].index)
        tests_to_drop = [k for sublist in [[i for i in columns if j in i] for j in tests_to_drop] for k in sublist]

        df_final = df_final.drop(tests_to_drop, axis=1)

        icds_to_drop = df_final[icd9_columns].sum().sort_values(ascending=True)[:-n_icd9_codes].index
        df_final = df_final.drop(icds_to_drop, axis=1)

        # Only consider those patients with diabetes who have not been diagnosed with the outcome
        # before baseline
        df_final = df_final.loc[list(ids_pre_diabetes_onset.keys())]

        # Drop ICD-9 buckets
        if not icd9_buckets:
            df_final = df_final.drop([i for i in df_final.columns if 'bucket' in i], axis=1)

        # Drop ICD-9 code 250
        if drop_icd9_250 and 'icd9_prefix=250' in df_final.columns:
            df_final = df_final.drop('icd9_prefix=250', axis=1)

    # Replace missing values of ICD-9 and medication data with 0.
    df_final = impute_icd9_med_data(df_final)
    
    # Remove outliers
    if outlier_removal:
        df_final = remove_outliers(df_final, lower, upper)
        
    # Add smoking predictor
    if add_smoking:
        df_final = smoking(df_final, df_icd)

    # Add predictor with time since diagnosis
    if time_since_diagnosis:
        df_final['time_since_diagnosis'] = np.repeat(baseline, len(ids_pre_diabetes_onset)) - \
                                           np.array(list(ids_pre_diabetes_onset.values()))

    # Add labels depending on forecast horizon (pred_year)
    df_final_indices = list(df_final.index.get_level_values('id').unique())
    df_final['y'] = [1 if i in ids_outcome_onset.keys() and 
                     ids_outcome_onset[i] <= baseline + pred_year 
                     else 0 for i in df_final_indices]
    
    return df_final

def generate_data(df_lab,
                  df_icd,
                  df_demo,
                  df_med,
                  antidiabetes_medication,
                  baseline,
                  pred_year,
                  population,
                  outcomes,
                  predictors=None,
                  store=False,
                  file_ending=None,
                  estimate_gfr=True,
                  alb_scr_level=30,
                  egfr_level=60,
                  include_lab_nephropathy=True,
                  egfr_method='ckd-epi', 
                  add_trajectory=False,
                  only_trend=False,
                  lab_value_method='mean',
                  date_type='year',
                  ffill_lab=True,
                  ffill_demo=True,
                  ffill_icd=True,
                  ffill_med=0,
                  add_smoking=False,
                  time_since_diagnosis=False,
                  n_icd9_codes=60,
                  n_tests=60,
                  icd9_buckets=False,
                  drop_icd9_250=True,
                  outlier_removal=False,
                  lower=0.001,
                  upper=0.999,
                  path=None):
    """
    Generate final dataset.
    """
    data_final_all = {}
    for outcome in outcomes:
        
        # Calculate IDs for patientes with diabetes
        ids_diabetes = helper.compute_ids_diabetes_onset(df_lab=df_lab,
                                                         df_icd=df_icd, 
                                                         df_demo=df_demo, 
                                                         df_med=df_med, 
                                                         antidiabetes_medication=antidiabetes_medication, 
                                                         outcome=outcome, 
                                                         baseline=baseline)
        
        # Define IDs of population
        if population == 'diabetes':
            ids_population = ids_diabetes
        elif population == 'prediabetes':
            ids_population = helper.compute_ids_prediabetes_onset(df_lab=df_lab, 
                                                                  df_icd=df_icd, 
                                                                  df_demo=df_demo, 
                                                                  ids_diabetes=ids_diabetes, 
                                                                  outcome=outcome, 
                                                                  baseline=baseline)
        else:
            raise ValueError('Unknown population. Choose either diabetes or prediabetes.')

        # Define ICD-9 codes of outcomes
        outcome_icd9_codes = helper.outcome_code(outcome)
        ids_population_outcome, ids_outcome = helper.compute_ids_outcome_onset(df_icd=df_icd, 
                                                                               df_lab=df_lab, 
                                                                               ids_pre_diabetes=ids_population,
                                                                               outcomes=outcome_icd9_codes,
                                                                               pred_year=pred_year,
                                                                               baseline=baseline,
                                                                               estimate_gfr=estimate_gfr,
                                                                               df_demo=df_demo,
                                                                               alb_scr_level=alb_scr_level,
                                                                               egfr_level=egfr_level,
                                                                               include_lab_nephropathy=include_lab_nephropathy,
                                                                               egfr_method=egfr_method)

        # Preprocess data
        data = preprocess_data(df_lab=df_lab,
                               df_demo=df_demo,
                               ids_pre_diabetes_onset=ids_population,
                               df_icd=df_icd,
                               df_med=df_med,
                               baseline=baseline,
                               add_trajectory=add_trajectory,
                               only_trend=only_trend,
                               lab_value_method=lab_value_method,
                               date_type=date_type,
                               ffill_lab=ffill_lab,
                               ffill_demo=ffill_demo,
                               ffill_med=ffill_med)

        data_final_all[outcome] = final_preprocessing_step(data=data,
                                                           ids_pre_diabetes_onset=ids_population_outcome,
                                                           ids_outcome_onset=ids_outcome, 
                                                           pred_year=pred_year, 
                                                           df_icd=df_icd,
                                                           baseline=baseline,
                                                           predictors=predictors,
                                                           add_trajectory=add_trajectory,
                                                           add_smoking=add_smoking,
                                                           time_since_diagnosis=time_since_diagnosis,
                                                           n_icd9_codes=n_icd9_codes,
                                                           n_tests=n_tests,
                                                           icd9_buckets=icd9_buckets,
                                                           drop_icd9_250=drop_icd9_250,
                                                           ffill_icd=ffill_icd,
                                                           outlier_removal=outlier_removal,
                                                           lower=lower,
                                                           upper=upper)
        
    if store:
        np.save(path+f'data_{population}_{file_ending}.npy', data_final_all)
        
    return data_final_all

def generate_lstm_data(population,
                       df_demo,
                       df_lab, 
                       df_icd,
                       df_med,
                       antidiabetes_medication,
                       outcomes,
                       baseline,
                       pred_year,
                       ffill_lab,
                       store,
                       file_ending,
                       path):
    
    data = {}
    for outcome in outcomes:
        
        # Get diabetes IDs
        ids_diabetes = helper.compute_ids_diabetes_onset(df_lab=df_lab,
                                                         df_icd=df_icd, 
                                                         df_demo=df_demo, 
                                                         df_med=df_med, 
                                                         antidiabetes_medication=antidiabetes_medication, 
                                                         outcome=outcome, 
                                                         baseline=baseline)
        
        if population == 'prediabetes':
            ids_population = helper.compute_ids_prediabetes_onset(df_lab=df_lab,
                                                                  df_icd=df_icd,
                                                                  df_demo=df_demo,
                                                                  ids_diabetes=ids_diabetes,
                                                                  outcome=outcome,
                                                                  baseline=baseline)
        else:
            ids_population = ids_diabetes
        
        # Get baseline IDs w/o complication and outcome IDs
        outcome_icd9_codes = helper.outcome_code(outcome)
        ids_population_outcome, ids_outcome = helper.compute_ids_outcome_onset(df_icd=df_icd, 
                                                                             df_lab=df_lab, 
                                                                             ids_pre_diabetes=ids_population,
                                                                             outcomes=outcome_icd9_codes,
                                                                             pred_year=pred_year,
                                                                             baseline=baseline,
                                                                             estimate_gfr=True,
                                                                             df_demo=df_demo)
        
        # Preprocess lab data
        lab_sub = df_lab.loc[list(ids_population_outcome.keys())].copy()
        lab_prep = preprocessing_lab_df(lab_sub, baseline=baseline, ffill=ffill_lab)
        lab_prep = lab_prep.loc[pd.IndexSlice[:, :baseline], :]
        
        # Add missing years and select diabetes patients
        years = np.arange(2003, baseline+1)
        ids = list(lab_prep.index.get_level_values('id').unique())
        new_index = []
        for idx in ids:
            for year in years:
                new_index.append((idx, year))
        lab_prep = lab_prep.reindex(new_index)
        lab_prep = lab_prep.loc[ids_population_outcome]
        if ffill_lab:
            lab_prep = lab_prep.groupby(axis=0, level='id').ffill()
        
        # Add demo data
        df = lab_prep.join(df_demo.groupby(axis=0, level=['id', 'year']).mean())
        df['sex'] = df.groupby('id').sex.ffill().bfill()
        df['birthyear'] = df.groupby('id').birthyear.ffill().bfill()
        df['age'] = df.index.get_level_values('year') - df['birthyear']
        df.drop('birthyear', axis=1, inplace=True)
        
        # Add ICD-9 data
        icd_sub = df_icd.loc[list(ids_population_outcome.keys())].copy()
        icd_prep = preprocessing_icd9_df(icd_sub)
        df = df.join(icd_prep, how='left')
        
        # Add med data
        med_ids = list(df_med.index.get_level_values('id').unique())
        med_ids = [i for i in ids_population_outcome.keys() if i in med_ids]
        med_sub = df_med.loc[med_ids].copy()
        med_prep = preprocessing_med_df(med_sub)
        df = df.join(med_prep, how='left')
                
        # Replace nans with zeros
        df = df.replace(np.nan, 0)
                
        # Choose same predictors as for standard approach
        data_ = np.load(f'/local/home/sschallmoser/complications/data/data_{population}_basic.npy', allow_pickle=True).flatten()[0]
        cols = data_['eyes'].drop('y', axis=1).columns
        df = df[cols].copy()
        
        # Add y
        indices = df.index.get_level_values('id').unique()
        y = [1 if i in ids_outcome.keys() and 
                             ids_outcome[i] <= baseline + pred_year 
                             else 0 for i in indices]
        y = pd.DataFrame(y, index=indices)
        y['year'] = baseline
        y = y.reset_index().set_index(['id', 'year'])
        df['y'] = y
        data[outcome] = df.copy()
        
    if store:
        np.save(path+f'data_{population}_{file_ending}.npy', data)

    return data