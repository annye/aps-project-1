
from itertools import product 
import lightgbm as lgb
import numpy as np
import pandas as pd
from pdb import set_trace
import random
from sklearn.metrics import confusion_matrix as cm


def lgbm_multiclass(X_train, X_test, y_train, y_test, metrics_dict, key):
    """Trinary lgbm classifier."""

    # LGBM Dataset - create Dataset object for lgbm training
    lgtrain = lgb.Dataset(X_train, y_train, categorical_feature= "auto")
    lgbm_params =  {"task": "train",
                    "boosting_type": "gbdt",
                    "objective": "multiclass",
                    "num_class": 3,
                    "metric": ["multi_error"],
                    "learning_rate": 0.05,
                    'num_leaves': 2,
                    'min_data_in_leaf': 10,
                    'max_depth': 1,
                    'subsample': 0.3,
                    'colsample_bytree': 0.3
                    #'bagging_fraction': [0.75, 0.95]
                    }
    
    learning_rates = [0.05, 0.1]
    num_leaves = [2, 8, 16]
    min_data_in_leaf = [10, 20, 50, 100]
    max_depth = [1, 3, 5, 7, 9, 13]
    subsample = [0.3, 0.5, 0.8]
    colsample_bytree = [0.3, 0.5, 0.8]
    # Create all permutations for Random search
    hyp_params = [learning_rates, num_leaves, min_data_in_leaf,
                    max_depth, subsample, colsample_bytree]
    permutations = list(product(*hyp_params))
    
    # Randomly select ~5% of permutations
    #random.seed(42)
    rand_indices = random.sample(range(0, len(permutations)), int(np.floor(len(permutations) / 10)))
    rand_params = [permutations[x] for x in rand_indices]
    
    results = []
    for rand_param in rand_params:
    #for lr in learning_rates:
        #for nl in num_leaves:        
         #   for min_leaf in min_data_in_leaf:
          #      for md in max_depth:
           #         for sub in subsample:
            #            for col_samp in colsample_bytree: 
                            lgbm_params['learning_rate'] = rand_param[0]
                            lgbm_params['num_leaves'] = rand_param[1]
                            lgbm_params['min_data_in_leaf'] = rand_param[2]
                            lgbm_params['max_depth'] = rand_param[3]
                            lgbm_params['subsample'] = rand_param[4]
                            lgbm_params['colsample_bytree'] = rand_param[5]
                            
						    # Find Optimal Parameters / Boosting Rounds
                            try:
	                            lgb_cv = lgb.cv(
	                                params = lgbm_params,
					                train_set = lgtrain,
									num_boost_round=2000,
									stratified=True,
									nfold = 10,
									verbose_eval=50,
									seed = 23,
									early_stopping_rounds=75
	                            )
                            except:
                            	set_trace()
                            loss = lgbm_params["metric"][0]                      
                            optimal_rounds = np.argmin(lgb_cv[str(loss) + '-mean']) 
                            best_cv_score = min(lgb_cv[str(loss) + '-mean'])
                            print("Optimal Round: {}, Optimal Score: {} + {}".format(optimal_rounds,best_cv_score,lgb_cv[str(loss) + '-stdv'][optimal_rounds]))                        
                            
                            results.append([key, optimal_rounds, best_cv_score,
                            	            lgb_cv[str(loss) + '-stdv'][optimal_rounds],
                            	            None, lgbm_params])
                            

    
    results_df = pd.DataFrame(results, columns=['Key', 'Optimal Rounds', 'Best CV Score (Loss)', 'Std Dev', 'LB', 'Parameters'])
    results_df.sort_values(by=['Best CV Score (Loss)'], ascending=True, inplace=True)
    best_params = results_df.iloc[0]['Parameters']
    best_nround = results_df.iloc[0]['Optimal Rounds']

    # Perform full training
    if best_nround == 0:
        model = lgb.train(best_params, lgtrain)
    else:
        model = lgb.train(best_params, lgtrain, num_boost_round=best_nround)

    # Predict on the test set
    pred = model.predict(X_test)
    pred_classes = np.argmax(pred, axis=1)
    confusion = cm(y_test, pred_classes)
    
    low_acc = confusion[0][0] / (confusion[0][0] + confusion[0][1] + confusion[0][2])
    med_acc = confusion[1][1] / (confusion[1][0] + confusion[1][1] + confusion[1][2])
    high_acc = confusion[2][2] / (confusion[2][0] + confusion[2][1] + confusion[2][2])
    
    metrics_dict[key]['low_acc'] = low_acc
    metrics_dict[key]['med_acc'] = med_acc
    metrics_dict[key]['high_acc'] = high_acc
    metrics_dict[key]['n_rounds'] = best_nround     
    return metrics_dict