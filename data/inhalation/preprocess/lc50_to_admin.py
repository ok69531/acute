#%%
import openpyxl
import warnings

import pandas as pd
import numpy as np

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings("ignore")


#%%
'''
    data split
'''

data = pd.read_excel('tg403_lc50.xlsx')


len(data['CasRN'].unique())

data['unit'].unique()
data['unit'].isna().sum()
data = data[data['unit'].notna()]
data = data[data['lower_value'].notna()]
# data = data[data['SMILES'].notna()]

casrn_na_idx = data[data['CasRN'] == '-'].index
# smiles_na_idx = data[data['SMILES'] == '-'].index

data = data.drop(casrn_na_idx).reset_index(drop = True)
# data = data.drop(list(casrn_na_idx) + list(smiles_na_idx)).reset_index(drop = True)


#%%
def unify(unit, value):
    if unit == 'mg/L':
        v_ = value
    
    elif unit == 'ppm':
        v_ = value
    
    elif unit == 'g/m^3':
        v_ = value
    
    elif unit == 'mg/m^3':
        v_ = value * 0.001
    
    elif unit == 'µg/m^3':
        v_ = value * 0.000001
    
    return v_


#%%
# gas data
lc50_gas_tmp = data[data['inhale type'] == 'gas']
lc50_gas_tmp['value'] = list(map(unify, lc50_gas_tmp.unit, lc50_gas_tmp.lower_value))
lc50_gas = lc50_gas_tmp.groupby(['Chemical', 'CasRN'])['value'].mean().reset_index()
# lc50_gas = lc50_gas_tmp.groupby(['CasRN', 'SMILES'])['time','value'].mean().reset_index()

len(lc50_gas) == len(lc50_gas.CasRN.unique())
len(lc50_gas) == len(lc50_gas.Chemical.unique())

# tqdm.pandas()
# lc50_gas['SMILES'] = lc50_gas.CasRN.progress_apply(lambda x: cirpy.resolve(x, 'smiles'))
# lc50_gas.SMILES.isna().sum()
# lc50_gas = lc50_gas[lc50_gas['SMILES'].notna()].reset_index(drop = True)

lc50_gas['category'] = pd.cut(lc50_gas.value, bins = [0, 100, 500, 2500, 20000, np.infty], labels = range(5))

lc50_gas.to_excel('../gas_wo_smiles.xlsx', header = True, index = False)


#%%
# vapour data
lc50_vap_tmp = data[data['inhale type'] == 'vapour']
lc50_vap_tmp['value'] = list(map(unify, lc50_vap_tmp.unit, lc50_vap_tmp.lower_value))
lc50_vap = lc50_vap_tmp.groupby(['CasRN'])['value'].mean().reset_index()
lc50_vap = pd.merge(lc50_vap, lc50_vap_tmp[['Chemical', 'CasRN']].drop_duplicates(subset = ['CasRN']), on = ['CasRN'], how = 'inner')

len(lc50_vap) == len(lc50_vap.CasRN.unique())
len(lc50_vap) == len(lc50_vap.Chemical.unique())

lc50_vap = lc50_vap[['Chemical', 'CasRN', 'value']]

# lc50_vap[lc50_vap.CasRN == '8006-64-2']
# lc50_vap[lc50_vap.CasRN == '93924-10-8']
# lc50_vap[lc50_vap.CasRN == '68551-17-7']
# lc50_vap[lc50_vap.CasRN == '1174522-20-3']

# lc50_vap['SMILES'] = lc50_vap.CasRN.progress_apply(lambda x: cirpy.resolve(x, 'smiles'))
# lc50_vap.SMILES.isna().sum()
# lc50_vap = lc50_vap[lc50_vap['SMILES'].notna()].reset_index(drop = True)

# lc50_vap = lc50_vap_tmp.groupby(['CasRN', 'SMILES'])['time', 'value'].mean().reset_index()
lc50_vap['category'] = pd.cut(lc50_vap.value, bins =[0, 0.5, 2.0, 10, 20, np.infty], labels = range(5))
lc50_vap.to_excel('../vapor_wo_smiles.xlsx', header = True, index = False)


#%%
# vapour data
lc50_aer_tmp = data[data['inhale type'] == 'aerosol']
lc50_aer_tmp['value'] = list(map(unify, lc50_aer_tmp.unit, lc50_aer_tmp.lower_value))
lc50_aer = lc50_aer_tmp.groupby(['CasRN'])['value'].mean().reset_index()
lc50_aer = pd.merge(lc50_aer, lc50_aer_tmp[['Chemical', 'CasRN']].drop_duplicates('CasRN'), on = ['CasRN'], how = 'left')

len(lc50_aer) == len(lc50_aer.CasRN.unique())
len(lc50_aer) == len(lc50_aer.Chemical.unique())

lc50_aer = lc50_aer[['Chemical', 'CasRN', 'value']]

# lc50_aer.CasRN.value_counts()
# lc50_aer[lc50_aer.CasRN == '28182-81-2']
# lc50_aer[lc50_aer.CasRN == '53880-05-0']
# lc50_aer[lc50_aer.CasRN == '99607-70-2']

# lc50_aer['SMILES'] = lc50_aer.CasRN.progress_apply(lambda x: cirpy.resolve(x, 'smiles'))
# lc50_aer.SMILES.isna().sum()
# lc50_aer = lc50_aer[lc50_aer['SMILES'].notna()].reset_index(drop = True)

lc50_aer['category'] = pd.cut(lc50_aer.value, bins =[0, 0.05, 0.5, 1.0, 5.0, np.infty], labels = range(5))

lc50_aer.to_excel('../aerosol_wo_smiles.xlsx', header = True, index = False)

