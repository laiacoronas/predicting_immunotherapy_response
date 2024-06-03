import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import numpy as np
import json
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score
import shap
from sklearn.impute import KNNImputer

#ML
def lasso_fselection(X_train, y_train, inner_cv):
    pipe = Pipeline([
        ('selector', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', max_iter=500, random_state=123))),
        ('classifier', LogisticRegression(penalty='elasticnet', solver='saga', max_iter=500))
    ])
    param_distributions = {
        'selector__estimator__C': [0.01, 0.1, 1, 10],  
        'classifier__l1_ratio': [0.3,0.5,1]
    }
    
    rs = RandomizedSearchCV(pipe, param_distributions=param_distributions, n_iter=30, scoring='roc_auc', cv=inner_cv)
    rs.fit(X_train, y_train.values.ravel())
    selected_features = X_train.columns[rs.best_estimator_.named_steps['selector'].get_support()]
    results = {
        'selected_features': selected_features,
        'estimator_C': rs.best_params_['selector__estimator__C'],
        'l1_ratio': rs.best_params_['classifier__l1_ratio']
    }
    return results

def flexible_nested_cv_optimized(X, y, outer_cv, inner_cv):

    inner_results = []
    outer_results = []
    
    outer_fold_counter = 1  

    for train_idx_outer, val_idx_outer in outer_cv.split(X, y):
        feature_counts = defaultdict(int)
        C_counts = defaultdict(int)
        l1_ratio_counts = defaultdict(int)
        
        inner_fold_counter = 1  

        for train_idx_inner, val_idx_inner in inner_cv.split(X.iloc[train_idx_outer], y.iloc[train_idx_outer]):
            X_train_inner, y_train_inner = X.iloc[train_idx_inner], y.iloc[train_idx_inner]
            X_val_inner, y_val_inner = X.iloc[val_idx_inner], y.iloc[val_idx_inner]
            
            results_inner = lasso_fselection(X_train_inner, y_train_inner, inner_cv)
            final_features = results_inner['selected_features']
         
            for feature in final_features:
                feature_counts[feature] += 1
            C_counts[results_inner['estimator_C']] += 1
            l1_ratio_counts[results_inner['l1_ratio']] += 1
            
            inner_model = LogisticRegression(penalty='elasticnet', solver='saga', C=results_inner['estimator_C'], 
                                             l1_ratio=results_inner['l1_ratio'], max_iter=500, random_state=123)
            
            inner_model.fit(X_train_inner[final_features], y_train_inner.values.ravel())
            y_pred_inner = inner_model.predict_proba(X_val_inner[final_features])[:, 1]
            auc_inner = roc_auc_score(y_val_inner, y_pred_inner)
            
            inner_results.append({
                'Fold': 'outer_' + str(outer_fold_counter)+'_inner_' + str(inner_fold_counter),
                'Features': final_features, 
                'C': results_inner['estimator_C'], 
                'l1_ratio': results_inner['l1_ratio'],
                'AUC': auc_inner
            })
        
            inner_fold_counter += 1  

        most_common_features = [feature for feature, _ in Counter(feature_counts).most_common(5)]  
        most_common_C = Counter(C_counts).most_common(1)[0][0]
        most_common_l1_ratio = Counter(l1_ratio_counts).most_common(1)[0][0]
        
        X_train_outer, y_train_outer = X.iloc[train_idx_outer], y.iloc[train_idx_outer]
        X_val_outer, y_val_outer = X.iloc[val_idx_outer], y.iloc[val_idx_outer]
        
        final_model = LogisticRegression(penalty='elasticnet', solver='saga', C=most_common_C, l1_ratio=most_common_l1_ratio, max_iter=500, random_state=123)
        final_model.fit(X_train_outer[most_common_features], y_train_outer.values.ravel())
        y_pred_outer = final_model.predict_proba(X_val_outer[most_common_features])[:, 1]
        auc_outer = roc_auc_score(y_val_outer, y_pred_outer)
        
        outer_results.append({
            'Fold': 'outer_fold_' + str(outer_fold_counter),  
            'Features': most_common_features,
            'C': most_common_C,
            'l1_ratio': most_common_l1_ratio,
            'AUC': auc_outer
        })
        
        outer_fold_counter += 1 

    return pd.DataFrame(inner_results), pd.DataFrame(outer_results)

def final_model(X, y, outer_results_df, odir):
    all_features = [feat for sublist in outer_results_df['Features'].tolist() for feat in sublist]
    most_common_features = [feature for feature, _ in Counter(all_features).most_common(5)]
    
    most_common_C = Counter(outer_results_df['C']).most_common(1)[0][0]
    most_common_l1_ratio = Counter(outer_results_df['l1_ratio']).most_common(1)[0][0]
    
    final_model = LogisticRegression(penalty='elasticnet', solver='saga', C=most_common_C, 
                                     l1_ratio=most_common_l1_ratio, max_iter=500)
    final_model.fit(X[most_common_features], y.values.ravel())
    
    coefficients = final_model.coef_[0]
    intercept = final_model.intercept_[0]
    
    avg_auc = outer_results_df['AUC'].mean()
    std_auc = outer_results_df['AUC'].std()
    
    results = {
        'average_auc': avg_auc,
        'std_dev_auc': std_auc,
        'hyperparameters': {
            'C': most_common_C,
            'l1_ratio': most_common_l1_ratio
        },
        'model_coefficients': dict(zip(most_common_features, coefficients)),
        'model_intercept': intercept,
        'features': most_common_features
    }
    
    name='clinical'
    plt.figure()
    plt.title('SHAP values for the clinical signature')
    X = X.fillna(0)
    explainer = shap.Explainer(final_model, X[most_common_features], feature_perturbation="interventional")
    shap_values = explainer.shap_values(X[most_common_features])
    shap.summary_plot(shap_values, X[most_common_features], feature_names=most_common_features)
    final_model_path = os.path.join(odir, name+".json")
    with open(final_model_path, 'w') as f:
       json.dump(results, f, indent=4)
    return final_model_path

def main(argv):
    cldir='C:\\Users\\lclai\\Desktop\\laia\\csvs\\train\\clinical_biological_db_20230905.csv'
    odir='C:\\Users\\lclai\\Desktop\\laia\\csvs\\'
    os.makedirs(odir, exist_ok=True)
    clin_df=pd.read_csv(cldir)
    clin_df = clin_df.filter(regex='^(?!Unnamed)')
    clindata=clin_df[['patient_id', 'age', 'liver_met', 'treatment', 'sex', 'primary_tlocation', 'cb_4mo',
                       "baseline_lymphos", "baseline_neutroph", "baseline_leucocytes", "baseline_platelets", "baseline_albumin", "baseline_ECOG"]]
    
    median_age = clindata['age'].median()
    clindata['age_above_median'] = (clindata['age'] > median_age).astype(int)
    clindata.drop('age', axis=1, inplace=True)
    
    # Selecting the columns for imputation
    columns_for_imputation = [
        "baseline_lymphos", "baseline_neutroph", "baseline_leucocytes",
        "baseline_platelets", "baseline_albumin", "baseline_ECOG"]
    data_for_imputation = clindata[columns_for_imputation]
    # Using KNN imputer with n_neighbors=5
    knn_imputer = KNNImputer(n_neighbors=5)
    imputed_data = knn_imputer.fit_transform(data_for_imputation)
    
    # Creating a DataFrame for the imputed data
    imputed_data_df = pd.DataFrame(imputed_data, columns=columns_for_imputation)
    for column in columns_for_imputation:
        clindata.loc[:, column] = imputed_data_df[column]
    
    y = clindata['cb_4mo']
    patients = clindata[['patient_id']]
    toscale = clindata[['baseline_lymphos',
        'baseline_neutroph', 'baseline_leucocytes', 'baseline_platelets','baseline_albumin']]
    binary = clindata[['sex','treatment', 'liver_met', 'age_above_median']]
    categorical = clindata[['primary_tlocation', 'baseline_ECOG']]

    #scaling
    scaler = StandardScaler()
    scaled = scaler.fit_transform(toscale)
    scaled = pd.DataFrame(scaled, columns=toscale.columns)

    #encoding
    le = LabelEncoder()
    binary.loc[:, 'sex'] = le.fit_transform(binary['sex'])  # 1 es male i 0 es female
    binary.loc[:, 'treatment'] = binary['treatment'].str.strip()
    binary.loc[:, 'treatment'] = le.fit_transform(binary['treatment'])  # 0 combo 1 monotherapy
   
   # Threshold primary_tlocation and baseline_ECOG
    primarytumor = ['lung', 'colorectal', 'skin', 'breast', 'cervix', 'head_and_neck']
    ecog_levels = [0, 1, 2, 3, 4]
    for i in range(len(categorical)):
        if categorical.loc[i, 'primary_tlocation'] not in primarytumor:
            categorical.loc[i, 'primary_tlocation'] = 'other'

    # Encoding primary_tlocation
    dummies_primary_tlocation = pd.get_dummies(categorical['primary_tlocation'], prefix='encod_primary_tlocation')
    dummies_primary_tlocation = dummies_primary_tlocation - 0.5

    # Encoding baseline_ECOG
    dummies_baseline_ECOG = pd.get_dummies(categorical['baseline_ECOG'], prefix='encod_baseline_ECOG')
    dummies_baseline_ECOG = dummies_baseline_ECOG - 0.5

    Xdef = pd.concat([scaled, binary, dummies_primary_tlocation, dummies_baseline_ECOG], axis=1)
    Xdefy = pd.concat([patients, y,scaled, binary, dummies_primary_tlocation, dummies_baseline_ECOG], axis=1)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    inner_results_df, outer_results_df=flexible_nested_cv_optimized(Xdef, y, outer_cv, inner_cv)
    final_results = final_model(Xdef, y, outer_results_df, odir)
    print("Finish!! Results are saved here: ", odir)
    
if __name__=="__main__":
    main(sys.argv[1:])
    print("Finish!")

