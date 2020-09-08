from all_imports import * 

class RandomForest() :
    """
  
    """

    def __init__(self) :
        """This class perform Random Forrest Classifier """

        print (" **RandomForrest** Object created")
    

    def rf_classifier(self, X_train, X_test, y_train, y_test, metrics_dict, key):
        """ Run cross validation on train datasets.
    
        """
    
        
        parameter_grid = {'bootstrap': [True, False],
                      'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                      'max_features': ['auto', 'sqrt'],
                      'min_samples_leaf': [1, 2, 4],
                      'min_samples_split': [2, 5, 10],
                      'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
                      }
       
        
                       
        # # Create classifier object
        clf = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                                  param_distributions=parameter_grid,
                                  cv=10,
                                  n_iter=1,
                                  n_jobs=-1,
                                  random_state=42,
                                  scoring='roc_auc'
                                  )
        #clf = GridSearchCV(estimator=LGBMClassifier(random_state=42), param_grid=parameter_grid, cv=10, n_jobs=-1)
        # Train the classifier
        clf.fit(X_train, y_train)  
        
        # Train a new classifier on all the cross-validation data using the best parameters found by grid search
        clf_full_training = RandomForestClassifier(random_state=42,
                                           #class_weights='balanced',      
                                           bootstrap=clf.best_params_['bootstrap'],
                                           max_depth=clf.best_params_['max_depth'],
                                           max_features=clf.best_params_['max_features'],
                                           min_samples_leaf=clf.best_params_['min_samples_leaf'],
                                           min_samples_split=clf.best_params_['min_samples_split'],
                                           n_estimators=clf.best_params_['n_estimators']).fit(X_train, y_train)
                                        
        
        #clf_full_training = RandomForestClassifier(random_state=42)
        #clf_full_training.fit(X_train, y_train)        

        # Test the optimised classifier on the hold out test set
        print(clf_full_training.score(X_test, y_test))
        y_pred = clf_full_training.predict(X_test)
        
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

        #sns.heatmap(cm, annot=True)
        
        # Get confusion using custom function
        #plot_confusion_matrix(y_test,
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
        #plt.figure()
        #lw = 2
        #plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
        #plt.plot(fpr, tpr, color='darkorange',
        #        lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        #plt.ylim([0.0, 1.05])
        ##plt.xlim([0.0, 1.0])
        #plt.xlabel('False Positive Rate')
        #plt.ylabel('True Positive Rate')
        #plt.title('Receiver operating characteristic')
        #plt.legend(loc="lower right")
        #plt.show()
        return metrics_dict
        
