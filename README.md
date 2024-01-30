# Predicting_immunotherapy_response

This project aims to predict the outcome of oncologic patients with clinical and image data. Image features are acquired through the segmentation and feature extraction with PyRadiomics of computed tomographic images of solid tumors. The end-point considered is the response or unresponse of the immunotherapy according to RECIST criteria at 5 month from the starting date of the treatment.

The project can be subdivided into diferent steps:

1. Determining the optimal radiomic workflow
   a. Best feature aggregarion method (we have several leasions for each of the patients)
       - Largest lesion
       - Unweighted mean
       - Weighted mean of all lesions
       - Weighted mean of the three largest lesions
   b. Best feature selector (which output is the best features to predict the immunotherapy response)
       - LASSO
       - MRMR
       - ANOVA
   C. Best classifier
       - Logistic regression
       - Random Forest
       - XGBoost
All the results are generated computing the mean AUC of each fold in a nested cross validation approach. In addition, in each fold we save the results for the best hyperparameters and most predictive features, generating therefore a dictionary of counts.

2. Comparing the results between the best algorithm in the above-mentioned section and the results obtained using clinical or combined (clinical + radiomic data) signatures. This allows as to know if either clinical or radiomic features add predictive power to the model.
3. Accurately visualizing the results, performing statistical tests to identify significant differences and providing explainability to the model (3D maps + shap values).
