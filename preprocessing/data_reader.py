import pandas as pd
import numpy as np
from categories_dicts import bucketize_icds, test_name_to_bucket, icd9bucket_to_name, med_to_categories, med_to_class

path = '/local/home/sschallmoser/data/'

def read_demo_data(date_type='year'):
    
    print('Read demo data')
    
    # Read data
    df_bmi = pd.read_csv(path+'INFO_BMI.csv', encoding='latin-1', header=None)
    df_bp = pd.read_csv(path+'INFO_BP.csv', encoding='latin-1', header=None)
    df_demo = pd.read_csv(path+'INFO_DEMO.csv', encoding='latin-1', header=None)
    
    # Define date index depending on type
    if date_type == 'year':
        date_index = 4
    elif date_type == 'date':
        date_index = 8
    else:
        raise ValueError('Unknown date type.')    
    
    # Clean BMI data
    df_bmi['year'] = df_bmi[1].apply(lambda row: int(str(row)[:date_index]))
    df_bmi = df_bmi.rename({0: 'id', 4: 'bmi'}, axis=1)
    df_bmi = df_bmi.set_index(['id', 'year'])
    
    # Clean blood pressure data
    df_bp['year'] = df_bp[1].apply(lambda row: int(str(row)[:date_index]))
    df_bp = df_bp.rename({0: 'id', 2: 'systolic_bp', 3: 'diastolic_bp'}, axis=1)
    df_bp = df_bp.set_index(['id', 'year'])
    df_bp = df_bp[['systolic_bp', 'diastolic_bp']]
    
    # Clean demo data
    df_demo['birthyear'] = df_demo[2].apply(lambda row: int(str(row)[:4]))
    df_demo['sex'] = df_demo[1] - 1
    df_demo = df_demo.rename({0: 'id'}, axis=1)
    df_demo.index = df_demo['id']
    df_demo = df_demo[['birthyear', 'sex']]
    df_demo = df_demo.groupby('id').first()
    
    # Join all three dataframes
    df = df_bmi.join(df_bp, how='outer')
    df = df.join(df_demo)
    
    df.index = pd.MultiIndex.from_tuples(df.index)
    df.index.names = ['id','year']
    df = df[['bmi', 'systolic_bp', 'diastolic_bp', 'birthyear', 'sex']]
    
    return df 

def read_icd9_data(date_type='year'):
    
    print('Read ICD-9 data')
    
    # Read data
    df = pd.read_csv(path+'INFO_AVH.csv', encoding='latin-1', 
                     header=None, usecols=[0, 1, 3])
    
    # Define date depending on type
    if date_type == 'year':
        df['year'] = df[1].apply(lambda x: int(str(x)[:4]))
    elif date_type == 'date':
        df['year'] = df[1].apply(lambda x: int(str(x)[:8]))
    else:
        raise ValueError('Unknown date type.')
    
    # Clean data
    df['id'] = df[0]
    df['icd9'] = df[3]
    df = df[df.icd9.apply(lambda x: x[0].isdigit())]
    df['icd9'] = df.icd9.apply(lambda x: x.replace(' ',''))
    df['icd9_prefix'] = df.icd9.apply(lambda x: int(x.split('.')[0]))
    df = df[['id', 'icd9', 'icd9_prefix', 'year']]
    df['icd9_bucket'] = df.icd9.apply(lambda x: bucketize_icds(x))
    df['icd9'].replace('907.O', '907.0', inplace=True)   
    df = df.set_index(['id', 'year'])
    
    return df

def read_med_data(date_type='year'):
    
    print('Read med data')
    
    # Define some helper functions
    def is_ascii(s):
        return all(ord(c) < 128 for c in s)
    
    def remove_amounts(s):
        return ' '.join([w for w in s.split() if not w[0].isdigit()])
    
    # Read data
    df = pd.read_csv(path+'INFO_MEDC.csv', encoding='latin-1', header=None)
    
    # Define date depending on type
    if date_type == 'year':
        df['year'] = df[4].apply(lambda x: int(str(x)[:4]))
    elif date_type == 'date':
        df['year'] = df[4].apply(lambda x: int(str(x)[:8]))
    else:
        raise ValueError('Unknown date type.')
    
    # Clean data
    df = df[[0, 2, 'year']]
    df.columns = ['id', 'med', 'year']
    df = df[df.med.apply(lambda x: is_ascii(x))]
    df['med'] = df.med.apply(lambda x: remove_amounts(x))
    df['med'] = df.med.apply(lambda x: x.lower())
    df['med'] = df.med.apply(lambda x: x.replace('.', ''))
    df = df[df.med.apply(lambda x: True if len(x.split()) >= 1 else False)]
    df['med'] = df.med.apply(lambda x: x.split()[0]) # Most drugs are described by first word
    df = df.drop_duplicates()    
    df = df.set_index(['id', 'year'])
    
    return df

def read_lab_data(date_type='year'):
    
    print('Read lab data')
    
    # Read data
    df = pd.read_csv(path+'INFO_LABF.csv', encoding='latin-1', 
                     header=None, usecols = [0, 1, 3, 4], 
                     dtype={0: 'int64', 1: 'int32',
                     3: str, 4: 'float32'})
    
    # Define date depending on type
    if date_type == 'year':
        df['year'] = df[1].apply(lambda x: int(str(x)[:4])).astype('int16')
    elif date_type == 'date':
        df['year'] = df[1].apply(lambda x: int(str(x)[:8])).astype('int32')
    else:
        raise ValueError('Unknown date type.')
        
    # Clean data
    df['id'] = df[0]
    df['test'] = df[3]
    df['test_outcome'] = df[4]
    df.test = df.test.apply(lambda x: x.lower())
    df.test = df.test.apply(lambda x: x.replace('\'',''))
    df.test = df.test.apply(lambda x: x.replace(' ',''))
    df['test_outcome'] = df.test_outcome.replace(0.0, np.nan)
    df = df.dropna()
    df = df[['id', 'test', 'test_outcome', 'year']]
    df = df.set_index(['id', 'year'])

    return df