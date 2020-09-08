from all_imports import * 
from data_reader import GetData
from collections import Counter


class Preprocess(GetData) :
    """
    Preprocess strategies defined and exected in this class
    """

    def __init__(self) :
        """This class perform  """

        print (" **Data process ** Object created")

    def binary_data(self, targets, class_num) :
        " Divides data and binarise values on target"
        techniques = targets.columns
        #Transforms the scores in a high/low scale"
        for tech in techniques:
            if class_num == 2: 
                criteria = [targets[tech].between(0, 5), targets[tech].between(6, 10)]
                values = [0, 1]
                targets[tech] = np.select(criteria, values)
            elif class_num == 3:
                criteria = [targets[tech].between(0, 3), targets[tech].between(4, 7), targets[tech].between(8, 10)]
                values = [0, 1, 2]
                targets[tech] = np.select(criteria, values)

        return targets, techniques

    def get_X_y(self, data, tech):
        # X = data.loc[:, "age" : "philosophy35"]
        
        X = data[[
                'age','gender_Female','gender_Male','gender_Other','gender_Prefer not to say','country_Argentina',
                'country_Australia','country_Bangladesh','country_Canada','country_Chile','country_China - Hong Kong / Macau',
                'country_Denmark','country_England/United Kingdom','country_Estonia','country_Finland','country_France','country_Germany',
                'country_Ghana','country_Great Britain','country_Greece','country_Ireland','country_Israel and the Occupied Territories',
                'country_Italy','country_Netherlands','country_New Zealand','country_Nigeria','country_Philippines','country_Poland',
                'country_Portugal','country_Romania','country_Slovenia','country_South Africa','country_Spain','country_Sri Lanka',
                'country_United States of America (USA)','country_Virgin Islands (UK)','country_Virgin Islands (US)','education_Bachelors Degree',
                'education_High School','education_Masters Degree','education_Ph.D. or higher','education_Prefer not to say',
                'education_Some High School','education_Trade School','extraversion_enthusiastic','critical_quarrelsome','dependable_self-disciplined',
                'anxious_easily_upset','openness_to_experiences','reserved_quiet','sympathetic_warm','disorganized_careless',
                'calm_emotionally_stable','conventional_uncreative','extroversion_dim','agreeableness_dim','conscientiousness_dim',
                'emotional_stability_dim','openness_dim', 'philosophy1' ,'philosophy2' ,'philosophy3' ,'philosophy4',
                'philosophy5' ,'philosophy6' ,'philosophy7','philosophy8' ,'philosophy9' ,'philosophy10','philosophy11',
                'philosophy12','philosophy13','philosophy14' ,'philosophy15'  ,'philosophy16','philosophy17','philosophy18',
                'philosophy19','philosophy20','philosophy21' ,'philosophy22','philosophy23','philosophy24',
                'philosophy25','philosophy26','philosophy27','philosophy28','philosophy29','philosophy30','philosophy31','philosophy32',
                'philosophy33','philosophy34','philosophy35','approval','love','achievement','perfectionism','entitlement','omnipotence','autonomy',
                'approval_level', 'love_level','achievement_level','perfectionism_level',
                'entitlement_level','autonomy_level','omnipotence_level','extroversion_level', 
                'agreeableness_level','conscientiousness_level','emotional_stability_level', 'openness_level']]

        y = tech
      
        return X, y

    def train_test(self, X, y):
        "Split data train and test"  
        try:
            X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.25, random_state = 33)
        except:
            set_trace()
        return X_train, X_test, y_train, y_test
        

    def standard_scaler(self, X_train, X_test) :
        """Scaling data before classifier."""
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        print ("Train feature shape:", X_train.shape)
        print("Train feature shape:", X_test.shape)
        return X_train, X_test
    

    def resampling_split(self, X, y, imb_key):
        counter = Counter(y)
        print("Data before sampling:\n{}",counter)
        if imb_key == 'rus':
            rus = RandomUnderSampler()
            X_rus, y_rus = rus.fit_sample(X, y)
            counter = Counter(y_rus)
            y_rus = y_rus.values.reshape(1,-1)
            print("Data after Over-sampling:\n{}",counter)
            return X_rus, y_rus     
        elif imb_key == 'ros':
            ros = RandomOverSampler()
            X_ros, y_ros = ros.fit_sample(X, y)
            counter = Counter(y_ros)
            y_ros = y_ros .values.reshape(1,-1)
            print("Data after Under-sampling:\n{}",counter)
            return X_ros, y_ros
        elif imb_key == 'smote':
            sm = SMOTE(random_state=42)
            X_smote, y_smote = sm.fit_sample(X, y)
            counter = Counter(y_smote)
            y_smote = y_smote.values.reshape(1,-1)
            print("Data after SMOTE:\n{}",counter)
            return X_smote, y_smote


    def standard_scaler_Resampled(self,
                                   X_RUS, y_RUS, X_ROS,y_ROS,X_SMOTE,
                                    y_SMOTE,X_RUS_train, X_RUS_test, y_RUS_train, 
                                    y_RUS_test, X_ROS_train, X_ROS_test, y_ROS_train, 
                                    y_ROS_test, X_SMOTE_train, X_SMOTE_test, y_SMOTE_train, y_SMOTE_test
                                    ) :
        """Scaling resampled data before classifier."""
        
        scaler = StandardScaler()
    
       
        X_RUS_train = scaler.fit(X_RUS_train)
        X_ROS_train = scaler.fit(X_ROS_train)
        X_SMOTE_train = scaler.fit(X_SMOTE_train)

        X_RUS_test = scaler.transform(X_RUS_test)
        X_ROS_test = scaler.transform(X_ROS_test)
        X_SMOTE_test = scaler.transform(X_SMOTE_test)
  
        return X_RUS_train, X_ROS_train,  X_SMOTE_train, X_RUS_test, X_ROS_test, X_SMOTE_test




    
