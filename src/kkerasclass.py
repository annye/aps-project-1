from math import sqrt
from time import time
import numpy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from all_imports import *
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow import keras
from tensorflow.keras import layers


class K_Class() :
    """
    will use Keras to define the model and class weights to help the model 
    learn from the imbalanced data. .
    """

    def __init__(self) :
      """
      Hyperparameter tuning
      """
      print('Keras NN -tuned parameters')
    
    def K_fit_model(self, X_train, X_test, y_train, y_test, metrics_dict, key):
      
      # Split the training set again to have a portion for determining
      # earlystopping point after hyperparameter tuning
      start = time.time()
      print(X_train.shape)
      print(X_test.shape)
      X_train_hyp, X_train_es, y_train_hyp, y_train_es = \
          train_test_split (X_train, y_train, test_size = 0.25, random_state = 33)
      
      y_train_es = to_categorical(y_train_es)
      y_train_hyp = to_categorical(y_train_hyp)
      y_test_pre_cat = y_test
      y_test = to_categorical(y_test)

      """
      X_train_hyp = X_train_hyp.iloc[0:20, :]
      X_train_es = X_train_es.iloc[0:20, :]
      X_test = X_test.iloc[0:20, :]
      y_train_hyp = y_train_hyp[0:20]
      y_train_es = y_train_es[0:20]
      y_test = y_test[0:20]
      y_test_pre_cat = y_test_pre_cat[0:20]
      """

      count_classes = y_test.shape[1]
      print(count_classes)
      # fix random seed for reproducibility
      seed = 7
      numpy.random.seed(seed)

      def create_model(optimizer='adam',
                       init_mode='uniform',
                       learn_rate=0.01,
                       momentum=0,
                       dropout=0.1):
        model = Sequential()
        model.add(Dense(500, activation='relu',kernel_initializer=init_mode, input_dim=X_train.shape[1]))
        model.add(Dropout(dropout))
        model.add(Dense(100, activation='relu',kernel_initializer=init_mode,))
        model.add(Dropout(dropout))
        model.add(Dense(50, activation='relu',kernel_initializer=init_mode,))
        model.add(Dense(2, activation='softmax',kernel_initializer=init_mode,))
        model.compile(loss='binary_crossentropy', 
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model

      model = KerasClassifier(build_fn=create_model,
                              epochs=100,
                              batch_size=10,
                              verbose=0
                              )

      # define the grid search parameters
      optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
      init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
      learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
      momentum =   [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
      dropout = [0.1, 0.3, 0.5]
      
      """
      optimizer = ['SGD']
      init_mode = ['uniform']
      learn_rate = [0.001]
      momentum =   [0.2]
      dropout = [0.1]
      """ 

      param_grid = dict(optimizer=optimizer,
                        init_mode=init_mode, 
                        learn_rate=learn_rate,
                        momentum=momentum)

      grid = RandomizedSearchCV(estimator=model,
                          param_distributions=param_grid,
                          n_iter=250,
                          n_jobs=-1,
                          cv=10
                          )
  
      # Use grid search
      grid_result = grid.fit(X_train_hyp, y_train_hyp)
      # summarize results
      print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
      means = grid_result.cv_results_['mean_test_score']
      stds = grid_result.cv_results_['std_test_score']
      params = grid_result.cv_results_['params']
      for mean, stdev, param in zip(means, stds, params):
          print("%f (%f) with: %r" % (mean, stdev, param))
      
      def plot_metric(history, metric):
          train_metrics = history.history[metric]
          val_metrics = history.history['val_'+metric]
          epochs = range(1, len(train_metrics) + 1)
          plt.plot(epochs, train_metrics)
          plt.plot(epochs, val_metrics)
          plt.title('Training and validation '+ metric)
          plt.xlabel("Epochs")
          plt.ylabel(metric)
          plt.legend(["train_"+metric, 'val_'+metric])
          plt.show()
      
      # patient early stopping - get epoch at which early stopping triggered
      es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
      es_model = create_model(optimizer=grid_result.best_params_['optimizer'],
                              init_mode=grid_result.best_params_['init_mode'],
                              learn_rate=grid_result.best_params_['learn_rate'],
                              momentum=grid_result.best_params_['momentum']
                              )
      history = es_model.fit(X_train_es, y_train_es, epochs=1000, validation_split=0.25, batch_size=10, verbose=2, callbacks=es)
      # Assess model performance using hold out test set 
      best_model = create_model(optimizer=grid_result.best_params_['optimizer'],
                                init_mode=grid_result.best_params_['init_mode'],
                                learn_rate=grid_result.best_params_['learn_rate'],
                                momentum=grid_result.best_params_['momentum']
                                )
      best_model.fit(X_train, y_train, epochs=es.stopped_epoch, batch_size=10, verbose=2)                          
      y_pred = best_model.predict_classes(X_test)
      
      # Get the confusion matrix using seaborn
      cm = confusion_matrix(y_test_pre_cat, y_pred)
      print(cm)
      sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])
      specificity = cm[0][0] / (cm[0][0] + cm[0][1])

      metrics_dict[key]['sens'] = sensitivity
      metrics_dict[key]['spec'] = specificity
      metrics_dict[key]['TP'] = cm[1][1]
      metrics_dict[key]['FP'] = cm[0][1]
      metrics_dict[key]['TN'] = cm[0][0]
      metrics_dict[key]['FN'] = cm[1][0]

      # Plot ROC curve and return AUC
      y_score = best_model.predict_proba(X_test)
      # Compute ROC curve and ROC area for each class
      fpr = dict()
      tpr = dict()
      roc_auc = dict()
      n_classes = 2
      fpr, tpr, _ = roc_curve(y_test_pre_cat, y_score[:,1])
      roc_auc = auc(fpr, tpr)
      metrics_dict[key]['auc'] = roc_auc
      print('AUC:  {}'.format(np.round(roc_auc, 3)))
      print(time.time() - start)
      set_trace()