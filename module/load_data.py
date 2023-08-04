import openpyxl
import pandas as pd
from .smiles2fing import smiles2fing


def load_data(admin_type, inhale_type = 'vapor'):
    if admin_type == 'inhalation':
        path = f'data/{admin_type}/{inhale_type}.xlsx'
    else:
        path = f'data/{admin_type}/{admin_type}.xlsx'
    
    df = pd.read_excel(path)
    drop_idx, fingerprints = smiles2fing(df.SMILES)
    
    y = df.category.drop(drop_idx).reset_index(drop = True)
    
    return fingerprints, y
