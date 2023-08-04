import re
import openpyxl
import pandas as pd

from rdkit import Chem
from rdkit.Chem import MACCSkeys


def smiles2fing(strings):
    mol_tmp = [Chem.MolFromSmiles(i) for i in strings]
    mol_none_idx = [i for i in range(len(mol_tmp)) if mol_tmp[i] == None]
    
    mol = list(filter(None, mol_tmp))
    
    maccs_tmp = [MACCSkeys.GenMACCSKeys(x) for x in mol]
    maccs = [re.split('', x.ToBitString(), maxsplit = 167)[1:] for x in maccs_tmp]
    
    fingerprints = pd.DataFrame(maccs).astype(int)
    fingerprints.columns = ['maccs' + str(i) for i in range(1, 168)]
    
    return mol_none_idx, fingerprints
