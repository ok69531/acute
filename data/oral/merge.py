import openpyxl
import pandas as pd

oral_desalt = pd.read_excel('TG420_desalt.xlsx')
# dermal_wo_smiles = pd.read_excel('oral420_wo_smiles.xlsx')

oral_desalt.columns = ['Chemical', 'CasRN', 'value', 'category', 'SMILES']
oral = oral_desalt[['Chemical', 'CasRN', 'SMILES', 'value', 'category']]

oral.to_excel('oral.xlsx', header = True, index = False)