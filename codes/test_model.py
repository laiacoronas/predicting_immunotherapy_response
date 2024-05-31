# Importing libraries
import os, sys, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, brier_score_loss,
    auc)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# Function to preprocess the data
def preprocess_data(rad_df, clin_df):
    clin_df = clin_df.filter(regex='^(?!Unnamed)')
    clin_df = clin_df.rename(columns={'predict_id': 'patient'})
    clin_df = clin_df.rename(columns={'n_loc_lesions': 'baseline_num_met_organs'})
    df=clin_df[['patient', 'treatment','age_above_median',  'sex', 'primary_tumour_localization','Leucocytes', 'Neutrophils', 'Lymphocytes', 'Platelets', 'LDH', 'Albumin',
       'n_lesions', 'baseline_num_met_organs', 'tumor_burden', 'liver_met',
       'cb_4mo']]
    
    #encoding
   # Encoding 'sex' column
    df['sex'] = df['sex'].replace({'Male': 1, 'Female': 0})

    # Encoding 'treatment' column
    df['treatment'] = df['treatment'].str.strip().replace({'combo immuno': 0, 'monotherapy': 1})
    
    # Define the mapping
    mapping = {
        "lung": "lung",
        "colon": "colorectal",
        "breast": "breast",
        "H&N": "head_neck",
        "ovarian": "ovary",
        "gastric": "upper_gi",
        "pancreatic": "others",
        "hepatocarcinoma": "others",
        "cholangiocarcinoma": "others",
        "endometrial": "others",
        "gastric": "upper_gi",
        "melanoma": "skin",
        "biliary_tract": "others",
        "rectal": "colorectal",
        "ureteral": "others",
        "mesothelioma": "others",
        "sacral chordoma": "others"
        }
    # Create the new column
    df.loc[:, "primary_tlocation"] = df["primary_tumour_localization"].map(mapping)
    df.drop('primary_tumour_localization', axis=1, inplace=True)
    categorical_cols = ['primary_tlocation']
    numerical_cols=['Leucocytes', 'Neutrophils', 'Lymphocytes', 'Platelets', 'LDH', 'Albumin',
       'n_lesions', 'tumor_burden', 'baseline_num_met_organs']
    
    
    
    # scaling numerical variabels, one-hot-encoding categorical variables 
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    df= pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df = df.rename(columns={'Neutrophils': 'baseline_neutrophils','Platelets': 'baseline_platelets', 'Albumin': 'baseline_albumin', 'Lymphocytes': 'baseline_lymphos'})
    print(df.columns)
    X_rad= rad_df.loc[:, ~rad_df.columns.str.contains('Unnamed')] #deleting unnamed column if any
    X_rad_scaled = scaler.fit_transform(X_rad.iloc[:, 1:])  # Don't scale 'patient' column
    X_rad_scaled = pd.DataFrame(X_rad_scaled, columns=X_rad.columns[1:])
    X_rad_scaled['patient'] = X_rad['patient'].values  # Add 'patient' back for merging
    df_final = pd.merge(df, X_rad_scaled, on="patient", how='inner')
    #df_final.to_csv(os.path.join(odir, 'ensemble_test.csv'), index=False)
    patient_ids = df_final['patient'].tolist()
    # Extract features (X) and target variable (y)
    X = df_final.drop(columns=['patient', 'cb_4mo'])
    # KNN imputation
    imputer = KNNImputer()
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)
    y = df_final['cb_4mo']
    return X, y, patient_ids

# Function to load the model coefficients, feature names and intercept from a JSON file
def load_model_coefficients(file_path):
    #Load the model coefficients and intercept from a JSON file.
    with open(file_path, 'r') as f:
        model = json.load(f)
        betas = model.get('model_coefficients', {})
        intercept = model.get('model_intercept', 0)
        feature_names = model.get('features', [])
    return betas, intercept, feature_names    

def logistic_function(r_name, X, betas, intercept, feature_names, y_true, patient_ids):
    # Ensure X contains only the features used in the model
    X = X[feature_names]

    # Convert betas from dictionary to array in the order of features in X
    beta_array = np.array([betas[feature] for feature in feature_names])

    # Calculate probabilities
    z = np.dot(X, beta_array) + intercept
    probabilities = 1 / (1 + np.exp(-z))
    
    # Map probabilities to patient IDs
    patient_probabilities = dict(zip(patient_ids, probabilities.tolist()))

    # ROC and AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve and AUC
    precision, recall, pr_thresholds = precision_recall_curve(y_true, probabilities)
    pr_auc = auc(recall, precision)

    # Brier Score
    brier = brier_score_loss(y_true, probabilities)

    # Confusion Matrix at optimal threshold using Youden's J statistic
    j_scores = tpr - fpr
    optimal_threshold = roc_thresholds[np.argmax(j_scores)]
    predictions = (probabilities > optimal_threshold).astype(int)
    conf_matrix = confusion_matrix(y_true, predictions)

    # Sensitivity and Specificity
    sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

    results = {
        "agg": r_name,
        "AUC-ROC": roc_auc,
        "AUC-PR": pr_auc,
        "Brier Score": brier,
        "Confusion Matrix": conf_matrix.tolist(),
        "Optimal Threshold": optimal_threshold,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        # "Predicted Probabilities": probabilities.tolist(),
        "Predicted Probabilities": patient_probabilities,
        # Add model details here
        "Model Coefficients": betas,
        "Model Intercept": intercept,
        "Feature Names": feature_names
    }

    return results


def main(argv):
# Loading testing  data
    clin_df=pd.read_csv(argv[0])
    rad_df=pd.read_csv(argv[1])
    X_scaled, y, patient_ids = preprocess_data(rad_df, clin_df)
    betas, intercept, feature_names = load_model_coefficients(argv[2])
    r_name=os.path.basename(argv[2]).split('/')[-1].split('.json')[0]
    results = logistic_function(r_name,X_scaled, betas, intercept, feature_names, y, patient_ids)
    odir=argv[3]
    
    # Export the results to a JSON file
    with open(odir+'/'+r_name+'_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # os.makedirs(odir, exist_ok=True)


if __name__=="__main__":
    main(sys.argv[1:])
    print("Finish!")

    #python nested_cv_vf.py largest lasso elasticnet
    
    #argv[0]: test clinical data
    #"/nfs/rnas/clolivia/_Experiments/immuno_paper/new_immuno_paper/final_csvs/test_predict_vhio/clin_vf_predict_test.csv"
    #argv[1]: test radiomics data
    # "/nfs/rnas/clolivia/_Experiments/immuno_paper/new_immuno_paper/final_csvs/test_predict_vhio/largest.csv"
    #argv[2]: trained model (json file)
    # '/nfs/rnas/clolivia/_Experiments/immuno_paper/new_immuno_paper/rad_results/weighted_3largest/results_weighted_3largest_elasticnet_lasso.json'
    #argv[3]: odir (where to store json file)

# testing the radiomcis model:
#    python test_model.py /nfs/rnas/clolivia/_Experiments/immuno_paper/laia/csvs/test/clinical_data_test.csv /nfs/rnas/clolivia/_Experiments/immuno_paper/laia/csvs/test/weighted_3largest.csv /nfs/rnas/clolivia/_Experiments/immuno_paper/laia/train_results/radiomics/3largest/results_3largest_logisticregression_lasso.json /nfs/rnas/clolivia/_Experiments/immuno_paper/laia/test_results

#  testing the clinical model:
#   python test_model.py /nfs/rnas/clolivia/_Experiments/immuno_paper/laia/csvs/test/clinical_data_test.csv /nfs/rnas/clolivia/_Experiments/immuno_paper/laia/csvs/test/weighted_3largest.csv /nfs/rnas/clolivia/_Experiments/immuno_paper/laia/train_results/clinical/clinical.json /nfs/rnas/clolivia/_Experiments/immuno_paper/laia/test_results
#  testing hte combined model
# python test_model.py /nfs/rnas/clolivia/_Experiments/immuno_paper/laia/csvs/test/clinical_data_test.csv /nfs/rnas/clolivia/_Experiments/immuno_paper/laia/csvs/test/weighted_3largest.csv /nfs/rnas/clolivia/_Experiments/immuno_paper/laia/train_results/combined/combined.json /nfs/rnas/clolivia/_Experiments/immuno_paper/laia/test_results

