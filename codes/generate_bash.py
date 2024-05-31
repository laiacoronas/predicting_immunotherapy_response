clf= ['logisticregression', 'xgb', 'randomf']
fsel=['lasso', 'mrmr', 'anova']
feat_agg=['largest', 'unweighted', '3largest', 'weighted']
code_path='/nfs/rnas/clolivia/_Experiments/immuno_paper/laia/codes/train_radiomics.py'

print("#!#!/usr/bin/bash") 
for c in clf:
    for f in fsel:
        for a in feat_agg:
            print ("python "+code_path+" {} {} {}".format(f,c,a))
#python generate_bash.py >> train_rad.sh

