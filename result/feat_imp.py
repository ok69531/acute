# https://zzinnam.tistory.com/entry/SHAP-value에-대한-간단한-소개with-Python
import shap

explainer = shap.Explainer(model, x)
shap_value = explainer(x)

shap.plots.waterfall(shap_value[0], max_display = 10)
shap.plots.beeswarm(shap_value)

shap_value.values
shap_value.base_values
shap_value.data

