#importing libraries
import os, sys
import pandas as pd
import numpy as np
import json
from collections import defaultdict, Counter
import argparse
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import mrmr
from mrmr import mrmr_classif
import shap


def fsel(X, y, sel, inner_cv):
    if sel=="lasso":
        selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', max_iter=500, random_state=59))
        param_distributions = {
            'estimator__C': [0.1, 1, 10]  # for Lasso
            }
        rs = RandomizedSearchCV(selector, param_distributions=param_distributions, n_iter=3, scoring='roc_auc', cv=inner_cv, random_state=59)
        rs.fit(X, y.values.ravel())
        #selected_features = X.columns[rs.best_estimator_.named_steps['selector'].get_support()]
        selected_features = X.columns[rs.best_estimator_.get_support()]

    elif sel=="anova":
        selector = SelectKBest(score_func=f_classif, k=5)
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]

    elif sel=="mrmr":
        selected_features = mrmr_classif(X, y, K=5)

    return selected_features
    
def classifier(X, y, cls, inner_cv, params=None):
    if params is not None:
        # If hyperparameters are provided, use them to define the classifier and skip tuning
        if cls == "logisticregression":
            return LogisticRegression(penalty='elasticnet', solver='saga', max_iter=500, random_state=123, 
                                      C=params['C'], l1_ratio=params['l1_ratio']), params
        
        elif cls == "randomf":
            return RandomForestClassifier(random_state=123, n_estimators=params['n_estimators'], 
                                          max_depth=params['max_depth']), params
        
        elif cls == "xgb":
            return xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=123, 
                                     n_estimators=params['n_estimators'], max_depth=params['max_depth'], 
                                     learning_rate=params['learning_rate'], subsample=params['subsample']), params
        
    else:    
        if cls=='randomf':
            classifier = RandomForestClassifier(random_state=123)
            param_distributions = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30]}

        elif cls=='xgb':
            classifier = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=123, n_jobs=-1)  # Stop early if no improvement after 10 rounds
            param_distributions = { #reduced range
                'n_estimators': [50, 75, 100],
                'max_depth': [3,5],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0],
            }
            
        elif cls=='logisticregression':
            classifier = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=500, random_state=123)
            param_distributions = {
                'C': [0.1, 1, 10],
                'l1_ratio': [0.3, 0.5, 1]
            }
        
        rs = RandomizedSearchCV(classifier, param_distributions=param_distributions, n_iter=27, 
                                    scoring='roc_auc', cv=inner_cv, verbose=3)
        rs.fit(X, y.values.ravel())
        return rs.best_estimator_, rs.best_params_

def process_most_common_hyperparams(hyperparam_counts):
    most_common_hyperparams_combined = dict(Counter(hyperparam_counts).most_common())
    processed_hyperparams = {}
    
    for combined_name, count in most_common_hyperparams_combined.items():
        name, value = combined_name.rsplit('_', 1)
        if name in ['n_estimators', 'max_depth']:
            try:
                processed_hyperparams[name] = int(value)
            except ValueError:
                # Handle or log the error if conversion fails
                pass
        else:
            try:
                processed_hyperparams[name] = float(value)  # Attempt to convert to float
            except ValueError:
                processed_hyperparams[name] = value  # If not possible, keep as string
    
    return processed_hyperparams

def nested_cv(X, y, outer_cv, inner_cv, cls, sel):
    inner_results = []
    outer_results = []

    # Outer loop
    outer_fold_counter = 1  

    for train_idx_outer, val_idx_outer in outer_cv.split(X, y):
        feature_counts = defaultdict(int)
        hyperparam_counts = defaultdict(int)  # To store hyperparameter counts from inner loop
        inner_fold_counter = 1  

        for train_idx_inner, val_idx_inner in inner_cv.split(X.iloc[train_idx_outer], y.iloc[train_idx_outer]):
            X_train_inner, y_train_inner = X.iloc[train_idx_inner], y.iloc[train_idx_inner]
            X_val_inner, y_val_inner = X.iloc[val_idx_inner], y.iloc[val_idx_inner]
        
            selected_features=fsel(X_train_inner, y_train_inner, sel, inner_cv)
            
            if len(selected_features) == 0:
                inner_results.append({
                    'Fold': f'outer_{outer_fold_counter}_inner_{inner_fold_counter}',
                    'Features': [], 
                    'AUC': None,
                    'Status': 'Failed due to 0 features selected'
                })
                continue
            
            
            # Classifier training and hyperparameter tuning in the inner loop
            best_classifier, classifier_params = classifier(X_train_inner[selected_features], y_train_inner, cls, inner_cv)
            
            # Update feature and classifier hyperparameter counts
            for feature in selected_features:
                feature_counts[feature] += 1
            for key, value in classifier_params.items():
                hyperparam_counts[key + "_" + str(value)] += 1
                
            # Train and evaluate model on inner loop using selected features and hyperparameters
            # Note: this is not really needed for obtaining the final model because we are not inerested in selecting the "best AUC" of the inner loop.
            # However, it's good to see what is going on in the inner loop, specially in the beginning to see if the outer results make sense
            # Evaluate model on inner loop
            y_pred_inner = best_classifier.predict_proba(X_val_inner[selected_features])[:, 1]
            auc_inner = roc_auc_score(y_val_inner, y_pred_inner)
            
            inner_results.append({
                'Fold': 'outer_' + str(outer_fold_counter)+'_inner_' + str(inner_fold_counter),
                'Features': selected_features, 
                'Classifier_Params': classifier_params,
                'AUC': auc_inner
            })
        
            inner_fold_counter += 1  # Increment the inner fold counter
        
        most_common_features = [feature for feature, _ in Counter(feature_counts).most_common(5)]
        most_common_hyperparams = process_most_common_hyperparams(hyperparam_counts)

        X_train_outer, y_train_outer = X.iloc[train_idx_outer], y.iloc[train_idx_outer]
        X_val_outer, y_val_outer = X.iloc[val_idx_outer], y.iloc[val_idx_outer]
        
        final_classifier,_=classifier(X_train_outer[most_common_features], y_train_outer, 
                                                cls,inner_cv,params=most_common_hyperparams)
        
        final_model=final_classifier.fit(X_train_outer[most_common_features], y_train_outer.values.ravel())
        y_pred_outer = final_model.predict_proba(X_val_outer[most_common_features])[:, 1]
        auc_outer = roc_auc_score(y_val_outer, y_pred_outer)
                
        outer_results.append({
            'Fold': 'outer_fold_' + str(outer_fold_counter),  # Using +1 for better naming consistency
            'Features': most_common_features,
            'Classifier_Params': most_common_hyperparams,
            'AUC': auc_outer
        })
        
        outer_fold_counter += 1  # Increment the inner fold counter

    return pd.DataFrame(inner_results), pd.DataFrame(outer_results)

class NumpyEncoder(json.JSONEncoder):  #we need this fro xgboost results. Error "object of type float32 is not JSON serializable"
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def get_results(X, y, outer_results_df, odir, agg, cls, sel):
    # Determine most frequently selected features from outer folds
    all_features = [feat for sublist in outer_results_df['Features'].tolist() for feat in sublist]
    most_common_features = [feature for feature, _ in Counter(all_features).most_common(5)]
    
   # Determine most frequently chosen hyperparameters from the outer results
    # Convert the column to string (this is likely redundant, but let's ensure it)
    outer_results_df['Params_String'] = outer_results_df['Classifier_Params'].astype(str)

    # Find the most common string representation of hyperparameters
    most_common_params_string = Counter(outer_results_df['Params_String']).most_common(1)[0][0]

    # Convert back to dictionary
    most_common_hyperparams = eval(most_common_params_string)    
        # Train a final model on all data using the most common features and hyperparameters

    if cls == "logisticregression":
        final_model = LogisticRegression(penalty='elasticnet', solver='saga', 
                                              C=most_common_hyperparams['C'], 
                                              l1_ratio=most_common_hyperparams['l1_ratio'], 
                                              max_iter=500, random_state=123)
    elif cls == "randomf":
        final_model = RandomForestClassifier(n_estimators=most_common_hyperparams['n_estimators'], 
                                                  max_depth=most_common_hyperparams['max_depth'], 
                                                  random_state=123)
    elif cls == "xgb":
        final_model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False, 
                                              n_estimators=most_common_hyperparams['n_estimators'], 
                                              max_depth=most_common_hyperparams['max_depth'], 
                                              learning_rate=most_common_hyperparams['learning_rate'], 
                                              random_state=123)
   
    final_model.fit(X[most_common_features], y.values.ravel()) #training with all my data... 
    
    # Calculate average and std deviation of AUCs from outer folds: to know estimated performance
    avg_auc = outer_results_df['AUC'].mean()
    std_auc = outer_results_df['AUC'].std()    
    
    # Prepare results for export
    results = {
        'classifier': cls,
        'selector': sel,
        'feature_aggregation': agg,
        'average_auc': avg_auc,
        'std_dev_auc': std_auc,
        'hyperparameters': most_common_hyperparams,
        'features': most_common_features
    }
    plt.figure()
    plt.title('SHAP values for the radiomics signature')
    explainer = shap.LinearExplainer(final_model, X[most_common_features], feature_perturbation="interventional")
    shap_values = explainer.shap_values(X[most_common_features])
    shap.summary_plot(shap_values, X[most_common_features], feature_names=most_common_features)
    # Extracting model details
    if cls == "logisticregression":
        results['model_coefficients'] = dict(zip(most_common_features, final_model.coef_[0]))
        results['model_intercept'] = final_model.intercept_[0]
    elif cls in ["randomf", "xgb"]:
        results['feature_importances'] = dict(zip(most_common_features, final_model.feature_importances_))
        
    final_model_path = os.path.join(odir, "results_"+agg+"_"+cls+"_"+sel+".json")
    with open(final_model_path, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder,  indent=4)

#main function
def main(argv):
    #define arguments, import csvs
    sel=argv[0]
    cls=argv[1]
    agg=argv[2]
    name=sel+'_'+cls
    rdir='C:\\Users\\lclai\\Desktop\\laia\\csvs\\train\\'+agg+'.csv'
    cdir='C:\\Users\\lclai\\Desktop\\laia\\csvs\\train\\clinical_biological_db_20230905.csv'
    odir='C:\\Users\\lclai\\Desktop\\laia\\train_results\\radiomics\\'+agg
    
    os.makedirs(odir, exist_ok=True)
    rad_df=pd.read_csv(rdir)
    clin_df=pd.read_csv(cdir)

    #preprocessing
    rad_df = rad_df.filter(regex='^(?!Unnamed)')
    rad_df = rad_df.rename(columns = {"Patient":"patient_id"})
    cb=clin_df[["patient_id","cb_4mo", "n_lesions", "baseline_num_met_organs", "total_seg_vol"]] #this has changed!
    cb=cb.dropna()
    rad_df = rad_df.dropna()
    df=pd.merge(cb, rad_df, on="patient_id", how='outer') 
    df = df.loc[:, ~df.columns.str.contains('Unnamed')] 
    df=df.drop(columns='original_firstorder_TotalEnergy')
    X, y=df.iloc[:,2:113], df.iloc[:,1:2]
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    X_scaled=pd.DataFrame(X_scaled,columns=X.columns)
    X_scaled = X_scaled.loc[y.index]

    #ML
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    inner_results_df,outer_results_df=nested_cv(X_scaled, y, outer_cv, inner_cv, cls, sel)
    inner_results_df.to_csv(os.path.join(odir, "inner_loop_"+agg+"_"+cls+"_"+sel+".csv"), index=False)
    outer_results_df.to_csv(os.path.join(odir, "outer_loop_"+agg+"_"+cls+"_"+sel+".csv"), index=False)
    final_results=get_results(X,y,outer_results_df, odir, agg, cls, sel)

if __name__=="__main__":
    main(sys.argv[1:])
    print("Finish!")
    
# To run the code use the following structure    
# python train_radiomics.py anova logisticregression largest

