import shap
import numpy as np
import pandas as pd

from module.load_data import load_data

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from scipy.stats import kendalltau

x, y = load_data('dermal')

def data_split(x, y, seed):
    num_test = round(len(x) * 0.1)
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = num_test, random_state = seed)
    # sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = seed)
    
    for train_idx, test_idx in sss.split(x, y):
        train_x = x.iloc[train_idx].reset_index(drop = True)
        train_y = y.iloc[train_idx].reset_index(drop = True)
        test_x = x.iloc[test_idx].reset_index(drop = True)
        test_y = y.iloc[test_idx].reset_index(drop = True)
    
    return train_x, test_x, train_y, test_y, num_test

seed = 0
splitseed = 0
x_train, x_test, y_train, y_test, num_val = data_split(x, y, splitseed)


# model = LogisticRegression()
model = RandomForestClassifier(random_state = seed)
model.fit(x_train, y_train)
pred = model.predict(x_test)

pd.crosstab(y_test, )

precision_score(y_test, pred, average = 'micro')
precision_score(y_test, pred, average = 'macro')

recall_score(y_test, pred, average = 'micro')
recall_score(y_test, pred, average = 'macro')

f1_score(y_test, pred, average = 'micro')
f1_score(y_test, pred, average = 'macro')

accuracy_score(y_test, pred)

kendalltau(y_test, pred).correlation


#
feat_imp = model.feature_importances_
top_idx = np.argsort(feat_imp)[::-1][:10]
feat_imp[top_idx]
x.columns[top_idx]

#
explainer = shap.TreeExplainer(model, x_train)
shap_value = explainer(x_train)

shap.plots.waterfall(shap_value[0][:, 0])

shap.initjs()
shap.plots.force(shap_value[0][:, 0])
shap.plots.force(shap_value[:, :, 0])

shap_value[0][:, 0]

shap.plots.beeswarm(shap_value[:, :, 0])

shap.plots.bar(shap_value[:, :, 0])

