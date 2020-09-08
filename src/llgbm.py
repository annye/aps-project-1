from all_imports import * 


class L_LGBM() :
    """
     LightGBM
    """

    def __init__(self) :
        """This class perform SVM """

        print (" **LGBM** Object created")
    

    def lgbm_classifier(self, X_train, X_test, y_train, y_test, metrics_dict, key):
        """ Run cross validation on train datasets.
    
        """
        
        parameter_grid  =  {'num_leaves': [2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8],
                            'min_data_in_leaf': [10, 20, 50, 75, 100, 300, 500, 700, 1000],
                            'max_depth': [1, 3, 5, 6, 7, 8, 9, 11, 13, 15, 17],
                            'subsample': [0.3, 0.5, 0.8],
                            'colsample_bytree': [0.3, 0.5, 0.8],
                            'bagging_fraction': [0.75, 0.95] ,
                            'boosting_type' : ['dart']
                            } 
        #parameter_grid =  {'num_leaves': [2**1],
        #                   'min_data_in_leaf': [10],
        #                   'max_depth': [1],
        #                   'subsample': [0.3],
        #                   'colsample_bytree': [0.3],
        #                   'bagging_fraction': [0.75],
        #                   'boosting_type' : ['dart'] 
        #                   } 

        
  
           
        # Create classifier object
        clf = RandomizedSearchCV(estimator=LGBMClassifier(random_state=42),
                                  param_distributions=parameter_grid,
                                  cv=10,
                                  n_iter=1,
                                  n_jobs=-1,
                                  random_state=42,
                                #   device = gpu,
                                #   gpu_platform_id = 0,
                                #   gpu_device_id = 0,
                                  scoring='roc_auc'
                                  )
        # clf = GridSearchCV(estimator=LGBMClassifier(random_state=42),
        #                                         param_grid=parameter_grid,
        #                                         cv=10,
        #                                         scoring='roc_auc',
        #                                          n_jobs=-1)
        # # Train the classifier
#################################################################
        
    

#################################################################
       
        clf.fit(X_train, y_train)  
        
        # # Train a new classifier on all the cross-validation data using the best parameters found by grid search
        clf_full_training = LGBMClassifier(random_state=42,
                                           class_weights='balanced',   
                                           num_leaves=clf.best_params_['num_leaves'],
                                           max_data_in_leaf=clf.best_params_['min_data_in_leaf'],
                                           max_depth=clf.best_params_['max_depth'],
                                           subsample=clf.best_params_['subsample'],
                                           colsample_bytree=clf.best_params_['colsample_bytree']
                                           ).fit(X_train, y_train)
             
        # Test the optimised classifier on the hold out test set
        print(clf_full_training.score(X_test, y_test))
        y_pred = clf_full_training.predict(X_test)

        f1 = f1_score(y_test, y_pred, average='macro')
        fw = f1_score(y_test, y_pred, average='weighted')
        bas = balanced_accuracy_score(y_test, y_pred)
        gm = geometric_mean_score(y_test, y_pred, average='weighted')
        ps = precision_score(y_test, y_pred,average='weighted')
        pre = metrics_dict.update( {'precision' : ps} )
        
        # Get the confusion matrix using seaborn
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
        specificity = cm[0][0] / (cm[0][0] + cm[0][1])

        metrics_dict[key]['sens'] = sensitivity
        metrics_dict[key]['spec'] = specificity
        metrics_dict[key]['TP'] = cm[1][1]
        metrics_dict[key]['FP'] = cm[0][1]
        metrics_dict[key]['TN'] = cm[0][0]
        metrics_dict[key]['FN'] = cm[1][0]
       
        metrics_dict[key].update( {'precision' : ps} )
        metrics_dict[key].update( {'BalAcc' : bas} )
        metrics_dict[key].update( {'geometric_mean_score' : gm} )
        metrics_dict[key].update( {'f1_weighted' : fw} )

       

        #sns.heatmap(cm, annot=True)
        
        # Get confusion using custom function
        # plot_confusion_matrix(y_test,
        #                    y_pred,
        #                    ['Low', 'High']
        #                    )
        # Plot ROC curve and return AUC
        y_score = clf_full_training.predict_proba(X_test)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = 2
        fpr, tpr, _ = roc_curve(y_test, y_score[:,1])
        roc_auc = auc(fpr, tpr)
        metrics_dict[key]['auc'] = roc_auc
        print('AUC:  {}'.format(np.round(roc_auc, 3)))
        return metrics_dict
        






