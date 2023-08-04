import openpyxl
import pandas as pd

dermal_desalt = pd.read_excel('TG402_desalt.xlsx')
dermal_wo_smiles = pd.read_excel('dermal402_wo_smiles.xlsx')

dermal_desalt.columns = ['CasRN', 'FOUND_BY', 'DTXSID', 'PREFERRED_NAME', 'SMILES']

dermal = pd.merge(dermal_desalt, dermal_wo_smiles[['CasRN', 'value', 'category']], how = 'left', on = ['CasRN'])

dermal.to_excel('dermal.xlsx', header = True, index = False)