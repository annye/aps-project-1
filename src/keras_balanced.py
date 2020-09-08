from all_imports import *
# from imblearn.keras import BalancedBatchGenerator
# from imblearn.tensorflow import balanced_batch_generator
# from balanced_batch_generator.tensorflow import balanced_batch_generator
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



class KerasIMB() :
    """
    Keras Imbalanced dataset
    """

    def __init__(self) :
        """This class perform SVM """

        print (" **LGBM** Object created")
    

    def nnk_classifier(self, X_train, X_test, y_train, y_test, metrics_dict, key):
       
        def convert_float64(X):
            return X.astype(np.float64)

        # numerical_columns = X_train
        # numerical_pipeline = make_pipeline(FunctionTransformer(func=convert_float64, validate=False),StandardScaler())
        # preprocessor = ColumnTransformer([('numerical_preprocessing',numerical_pipeline, numerical_columns), ('categorical_preprocessing')],remainder='drop')
      

        # Create a Neural Network

        def make_model(n_features):
            model = Sequential()
            model.add(Dense(200, input_shape=(n_features,),kernel_initializer='glorot_normal'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(100, kernel_initializer='glorot_normal', use_bias=False))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.25))
            model.add(Dense(50, kernel_initializer='glorot_normal', use_bias=False))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.15))
            model.add(Dense(25, kernel_initializer='glorot_normal', use_bias=False))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.1))
            model.add(Dense(1, activation='sigmoid'))

            model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

            return model

        def timeit(f):
            @wraps(f)
            def wrapper(*args, **kwds):
                start_time = time.time()
                result = f(*args, **kwds)
                elapsed_time = time.time() - start_time
                print('Elapsed computation time: {:.3f} secs'.format(elapsed_time))
                return (elapsed_time, result)
            return wrapper
        #    The first model will be trained using the fit method and with imbalanced mini-batches.

        @timeit
        def fit_predict_imbalanced_model(X_train, y_train, X_test, y_test,batch_size=len(X_test)):
                model = make_model(X_train.shape[1])
                model.fit(X_train,
                                y_train, epochs=2, verbose=1,batch_size= batch_size)
                y_pred = model.predict_proba(X_test, batch_size=batch_size)
                return roc_auc_score(y_test, y_pred)

        # In the contrary, we will use imbalanced-learn to create a generator of mini-batches which will yield balanced mini-batches.
        @timeit
        def fit_predict_balanced_model(X_train, y_train, X_test, y_test,batch_size=len(X_test)):
                X_train = np.asarray( X_train)
                model = make_model(X_train.shape[1])

                """  Error is here!, does not balanced_batch does not work, """
                training_generator = balanced_batch_generator(X_train, y_train,batch_size=batch_size,random_state=42)
                model.fit_generator(generator=training_generator, epochs=2, verbose=1)
                y_pred = model.predict_proba(X_test, batch_size=batch_size)
                return roc_auc_score(y_test, y_pred)

            #Classification loop¶
        skf = StratifiedKFold(n_splits=10)

        cv_results_imbalanced = []
        cv_time_imbalanced = []
        cv_results_balanced = []
        cv_time_balanced = []

        for train_idx, valid_idx in skf.split(X_train, y_train):
      
            scaler = StandardScaler()
            X_local_train = scaler.fit_transform(X_train).astype(float)
            X_local_test = scaler.transform(X_test).astype(float)
            y_local_train = y_train.values.ravel().astype(float)
            y_local_test = y_test.values.ravel().astype(float)


            X_local_train = np.asarray( X_local_train)
            X_local_test =  np.asarray( X_local_test)
            y_local_train = np.asarray( y_local_train)
            y_local_test = np.asarray( y_local_test)
     
  

            elapsed_time, roc_auc = fit_predict_imbalanced_model(X_local_train, y_local_train, X_local_test, y_local_test)
            cv_time_imbalanced.append(elapsed_time)
            cv_results_imbalanced.append(roc_auc)
            
        # set_trace()
        # #Plot the results and computational time
        # df_results = (pd.DataFrame({'Balanced model': cv_results_balanced,'Imbalanced model': cv_results_imbalanced}).unstack().reset_index())
        # # df_time = (pd.DataFrame({'Balanced model': cv_time_balanced,'Imbalanced model': cv_time_imbalanced}).unstack().reset_index())
         
        # df =  pd.DataFrame(list(zip(cv_results_balanced, cv_results_imbalanced)),columns=['Imbalanced_model', ['Balanced_model']))
        # df_time =  pd.DataFrame(cv_results_imbalanced, columns=['Imbalanced model'])

        #  dict = {'Balanced_model':cv_results_balanced,'Imbalanced model':cv_results_imbalanced}
        #  dfq = pd.DataFrame(dict) 
        #  df


        set_trace()


        # plt.figure()
        # sns.boxplot(y='level_0', x=0, data=df_time)
        # sns.despine(top=True, right=True, left=True)
        # plt.xlabel('time [s]')
        # plt.ylabel('')
        # plt.title('Computation time difference using a random under-sampling')

        # plt.figure()
        # sns.boxplot(y='level_0', x=0, data=df_results, whis=10.0)
        # sns.despine(top=True, right=True, left=True)
        # ax = plt.gca()
        # ax.xaxis.set_major_formatter(
        #     plt.FuncFormatter(lambda x, pos: "%i%%" % (100 * x)))
        # plt.xlabel('ROC-AUC')
        # plt.ylabel('')
        # plt.savefig('results/keras_model.png')
