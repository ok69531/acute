import openpyxl
import pandas as pd

vapor_desalt = pd.read_excel('TG403_vapor_desalt.xlsx')
aerosol_desalt = pd.read_excel('TG403_aerosol_desalt.xlsx')
gas_desalt = pd.read_excel('TG403_gas_desalt.xlsx')

vapor_wo_smiles = pd.read_excel('vapor_wo_smiles.xlsx')
aerosol_wo_smiles = pd.read_excel('aerosol_wo_smiles.xlsx')
gas_wo_smiles = pd.read_excel('gas_wo_smiles.xlsx')

vapor_desalt.columns = ['CasRN', 'FOUND_BY', 'DTXSID', 'PREFERRED_NAME', "SMILES"]
aerosol_desalt.columns = ['CasRN', 'FOUND_BY', 'DTXSID', 'PREFERRED_NAME', "SMILES"]
gas_desalt.columns = ['CasRN', 'FOUND_BY', 'DTXSID', 'PREFERRED_NAME', "SMILES"]

vapor = pd.merge(vapor_desalt, vapor_wo_smiles[['CasRN', 'value', 'category']], how = 'left', on = ['CasRN'])
aerosol = pd.merge(aerosol_desalt, aerosol_wo_smiles[['CasRN', 'value', 'category']], how = 'left', on = ['CasRN'])
gas = pd.merge(gas_desalt, gas_wo_smiles[['CasRN', 'value', 'category']], how = 'left', on = ['CasRN'])

vapor.to_excel('vapor.xlsx', header = True, index = False)
aerosol.to_excel('aerosol.xlsx', header = True, index = False)
gas.to_excel('gas.xlsx', header = True, index = False)