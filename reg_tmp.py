# generate fingerprints
# https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.htmlhttps://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html

#%%
import openpyxl
import numpy as np
import pandas as pd

from tqdm import tqdm
from inspect import getmembers, isfunction

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report
)

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator, MACCSkeys

from module.smiles2fing import smiles2fing
from module.argument import get_parser
from module.load_data import load_data

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300


#%%
data_tmp = pd.read_excel('data/dermal/dermal.xlsx')
# data_tmp = pd.read_excel('data/oral/oral.xlsx', index_col = 0)

mols_tmp = [Chem.MolFromSmiles(s) for s in data_tmp.SMILES]
none_idx = [i for i in range(len(mols_tmp)) if mols_tmp[i] == None]
mols = [mols_tmp[i] for i in range(len(mols_tmp)) if i not in none_idx]

data = data_tmp.drop(none_idx).reset_index(drop = True)

ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
mfp2gen = rdFingerprintGenerator.GetMorganGenerator(radius = 2)
fmgen = rdFingerprintGenerator.GetMorganGenerator(radius = 2, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()) # feature morgan fignerprints

fingerprints = pd.DataFrame([list(MACCSkeys.GenMACCSKeys(m)) for m in mols])
fingerprints = pd.DataFrame([list(ttgen.GetFingerprint(m)) for m in mols])
fingerprints = pd.DataFrame([list(mfp2gen.GetFingerprint(m)) for m in mols])
fingerprints = pd.DataFrame([list(fmgen.GetFingerprint(m)) for m in mols])

# data['weight'] = [Descriptors.ExactMolWt(mols[i]) for i in range(len(mols))]
# oral.value[oral.weight > 800]
# sum(oral.weight > 800)

descriptor_dict = {}
for x in getmembers(Descriptors, isfunction):
    if 'AUTOCORR2D' in x[0]:
        pass
    elif x[0] == '_ChargeDescriptors':
        break
    else:
        descriptor_dict[x[0]] = x[1]
        print(x[0])

exclude_descriptor_keys = [
    'BCUT2D_CHGHI',
    'BCUT2D_CHGLO',        
    'BCUT2D_LOGPHI',       
    'BCUT2D_LOGPLOW',      
    'BCUT2D_MRHI',         
    'BCUT2D_MRLOW',        
    'BCUT2D_MWHI',         
    'BCUT2D_MWLOW',        
    'MaxAbsPartialCharge', 
    'MaxPartialCharge',    
    'MinAbsPartialCharge', 
    'MinPartialCharge',
    'Ipc' # dermal
]

for k in exclude_descriptor_keys:
    descriptor_dict.pop(k, None)


descriptors = []
for i in tqdm(range(len(mols))):
    mol_descriptors = []
    for descriptor in descriptor_dict.keys():
        mol_descriptors.append(descriptor_dict[descriptor](mols[i]))
    descriptors.append(mol_descriptors)    

descriptors = pd.DataFrame(descriptors)
descriptors.columns = descriptor_dict.keys()

x = fingerprints
x = descriptors
x = pd.concat([fingerprints, descriptors], axis = 1)
y = data.category
y = data.value
# x.isna().sum()[x.isna().sum() != 0]

plt.figure(figsize = (7, 5))
plt.hist(y, bins = 50)
plt.title('Acute Dermal')
plt.show()
plt.close()

y.value_counts().sort_index()
y.value_counts(normalize = True).sort_index()


#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

model = RandomForestClassifier(random_state = 0)
model.fit(x_train, y_train)
pred = model.predict(x_test)

print('Confusion matrix')
print(pd.crosstab(pred, y_test, colnames = ['true'], rownames = ['pred']))
print('')
print('Metric')
print(classification_report(y_test, pred))


#%%
''' Classification '''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

model = RandomForestClassifier()
param_grid = {
    'n_estimators': [90, 100, 110, 120, 130, 140],
    # 'criterion': ['gini'],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2'],
    'random_state': [0]
}

grid_search = GridSearchCV(estimator = model, 
                           param_grid = param_grid,
                           cv = 5,
                           scoring = 'f1_macro')
grid_search.fit(x_train, y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)
best_model = grid_search.best_estimator_
pred = best_model.predict(x_test)

print('Confusion matrix')
print(pd.crosstab(pred, y_test, colnames = ['true'], rownames = ['pred']))
print('')
print('Metric')
print(classification_report(y_test, pred))


#%%
''' Regression '''
x = oral[descriptor_cols].to_numpy()
y = oral.value.to_numpy()
scaled_y = np.log(y)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_train, x_test, y_train, y_test = train_test_split(x, scaled_y, test_size = 0.2)

model = RandomForestRegressor()
param_grid = {'n_estimators': [5, 10, 30, 50, 100],
              'min_samples_split': [2, 4]}

grid_search = GridSearchCV(estimator = model,
                           param_grid = param_grid,
                           cv = 5)
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_
grid_search.best_params_
grid_search.best_score_

pred = best_model.predict(x_test)

mean_absolute_error(pred, y_test)
mean_squared_error(pred, y_test)

plt.scatter(pred, y_test)
plt.show()
plt.close()


# %%
