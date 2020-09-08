# coding=utf-8
"""
Create on 11th July
"""
import pandas as pd
import numpy as np 
from sklearn.preprocessing import OneHotEncoder


from pdb import set_trace


class GetData():
        """ Read csv file"""
        def __init__(self):
            """ Data Preparation"""
            print("Data preparation object created")

        def get_data(self) :
            """
            read data csv file
            """
           
            dataset= pd.read_csv('/Users/d18127085/Desktop/scripts_multiclass_lgbm/data/curated_text.csv')
           
            print ("Dataset shape:\n{}".format(dataset.shape))
            return dataset

        def feature_process(self,dataset):
            """TIPI  and DAS scale/scoring mapping & reverse-scored items calculation"""

            tipi= dataset[['extraversion_enthusiastic', 'critical_quarrelsome',
                        'dependable_self-disciplined', 'anxious_easily_upset',
                        'openness_to_experiences', 'reserved_quiet', 'sympathetic_warm',
                        'disorganized_careless','calm_emotionally_stable','conventional_uncreative']]
    
         
            value = {'Disagree strongly': '1', 
                         'Disagree moderately':'2',
                         'Disagree a little': '3',
                         'Neither agree nor disagree': '4',
                         'Agree a little': '5',
                         'Agree moderately': '6',
                         'Agree strongly':'7'}
            for v in value :
                    tipi = tipi.replace(value)

         
            tipi['extroversion_dim'] =tipi['extraversion_enthusiastic'].astype(float) +tipi['reserved_quiet'].astype(float)
            tipi['extroversion_dim'] =tipi['extroversion_dim']/ 2
            tipi['extroversion_level'] = ['0' if X <= 4.4 else "1" for X in tipi['extroversion_dim']]
                
            tipi['agreeableness_dim'] =tipi['critical_quarrelsome'].astype(float) +tipi['sympathetic_warm'].astype(float)
            tipi['agreeableness_dim'] = tipi['agreeableness_dim']/ 2
            tipi['agreeableness_level'] = ['0' if X <= 5.23 else "1" for X in tipi['agreeableness_dim']]

            tipi['conscientiousness_dim'] =tipi['dependable_self-disciplined'].astype(float) +tipi['disorganized_careless'].astype(float)
            tipi[ 'conscientiousness_dim'] =tipi[ 'conscientiousness_dim']/ 2
            tipi[ 'conscientiousness_level'] = ['0' if  X  <= 5.4 else '1' for X in tipi['conscientiousness_dim']]

            tipi['emotional_stability_dim'] =tipi['anxious_easily_upset'].astype(float) +tipi['calm_emotionally_stable'].astype(float)
            tipi['emotional_stability_dim'] = tipi['emotional_stability_dim']/ 2
            tipi['emotional_stability_level'] = ['0' if  X  <= 4.83 else '1' for X in tipi['emotional_stability_dim']]

            tipi['openness_dim'] =tipi['openness_to_experiences'].astype(float) +tipi['conventional_uncreative'].astype(float)
            tipi['openness_dim'] =tipi['openness_dim'] / 2
            tipi['openness_level'] = ['0' if  X  <= 5.38 else '1' for X in tipi['openness_dim']]

            # Personal Philosophy Subset

            values = dataset.loc[:,"philosophy1":"philosophy35"]
            scores = { 'Agree Strongly': '-2',
                        'Agree Slightly\t\t\t\t\t' : '-1',
                        'Neutral\t\t\t\t\t': '0',
                        'Disagree Slightly\t\t\t\t\t' : '1',
                        'Disagree Very Much' : '2'}

            for v in scores :
                values =values.replace(scores)
           

            approval =values.loc[:, 'philosophy1' : 'philosophy5'].astype(float)
            love =values.loc[:, 'philosophy6' :'philosophy10'].astype(float)
            achievement =values.loc[ :, 'philosophy11' :'philosophy15'].astype(float)
            perfectionism =values.loc[ :, 'philosophy16' : 'philosophy20'].astype(float)
            entitlement =values.loc[ :, 'philosophy21' : 'philosophy25'].astype(float)
            omnipotence =values.loc[ :, 'philosophy26' : 'philosophy30'].astype(float)
            autonomy =values.loc[ :, 'philosophy31' : 'philosophy35'].astype(float)

            columns1 = list(approval.columns)
            values['approval'] = approval[ columns1 ].sum (axis = 1)
            columns2 = list(love.columns)
            values['approval_level'] = ['0' if  X  <= 0 else '1' for X in values['approval']]
            values[ 'love' ] = love[ columns2 ].sum (axis = 1)
            values['love_level'] = ['0' if  X  <= 0 else '1' for X in values['love']]
            columns3 = list (achievement.columns)
            values[ 'achievement' ] = achievement[ columns3 ].sum (axis = 1)
            columns4 = list (perfectionism.columns)
            values['achievement_level'] = ['0' if  X  <= 0 else '1' for X in values['achievement']]
            values[ 'perfectionism' ] = perfectionism[ columns4 ].sum (axis = 1)
            columns5 = list (entitlement.columns)
            values['perfectionism_level'] = ['0' if  X  <= 0 else '1' for X in values['perfectionism']]
            values[ 'entitlement' ] = entitlement[ columns5 ].sum (axis = 1)
            columns6 = list (omnipotence.columns)
            values['entitlement_level'] = ['0' if  X  <= 0 else '1' for X in values['entitlement']]
            values[ 'omnipotence' ] = omnipotence[ columns6 ].sum (axis = 1)
            columns7 = list (autonomy.columns)
            values['omnipotence_level'] = ['0' if  X  <= 0 else '1' for X in values['omnipotence']]
            values['autonomy'] = autonomy[columns7].sum (axis = 1)
            values['autonomy_level'] = ['0' if  X  <= 0 else '1' for X in values['autonomy']]
    
            # Demographics set preparation
            demographics = dataset[['age', 'country', 'gender', 'education']]
            demo= pd.get_dummies(demographics, columns=['gender','country','education'], prefix=['gender','country','education'])

            techniques = dataset[['authority_edu', 'social_proof_edu', 'appeal_finances_edu',
                                    'awareness_words_edu', 'semantic_repetition_priming_edu',
                                    'appeal_to_common_sense', 'flattery_edu', 'hypocrisy_induction_edu',
                                    'scarcity_edu', 'device_rhetorical_question_edu',
                                    'device_hypophora_edu', 'semantic_repetition_edu', 'device_epistrophe',
                                    'device_anaphora_edu', 'device_antanagoge_edu',
                                    'Illusion_of_superiority_edu', 'appeal_finances_cs',
                                    'awareness_words_cs', 'authority_cs', 'social_proof_cs']]
            # New Dataset
            dataframes = [demo, tipi, values,techniques]
            data = pd.concat(dataframes, axis =1)
            data = pd.DataFrame(data)
            print ("++New Dataset shape is:\n{}".format(data.shape))
            return data
        

        def subset_data(self,data, subset):
            
            targets =data.loc[:,"authority_edu":"social_proof_cs"]
            if subset == 'all_ordinal':
                data = data[[   'age','gender_Female','gender_Male','gender_Other','gender_Prefer not to say',
                                'country_Argentina','country_Australia','country_Bangladesh','country_Canada',
                                'country_Chile','country_China - Hong Kong / Macau','country_Denmark','country_England/United Kingdom',
                                'country_Estonia','country_Finland','country_France','country_Germany','country_Ghana','country_Great Britain',
                                'country_Greece','country_Ireland','country_Israel and the Occupied Territories','country_Italy','country_Netherlands',
                                'country_New Zealand','country_Nigeria','country_Philippines','country_Poland','country_Portugal',
                                'country_Romania','country_Slovenia','country_South Africa','country_Spain','country_Sri Lanka',
                                'country_United States of America (USA)','country_Virgin Islands (UK)','country_Virgin Islands (US)',
                                'education_Bachelors Degree','education_High School','education_Masters Degree',
                                'education_Ph.D. or higher','education_Prefer not to say','education_Some High School',
                                'education_Trade School','extraversion_enthusiastic', 'critical_quarrelsome','dependable_self-disciplined',
                                'anxious_easily_upset','openness_to_experiences', 'reserved_quiet', 'sympathetic_warm',
                                'disorganized_careless', 'calm_emotionally_stable','conventional_uncreative',
                                'philosophy1', 'philosophy2', 'philosophy3', 'philosophy4',
                                'philosophy5', 'philosophy6', 'philosophy7', 'philosophy8',
                                'philosophy9', 'philosophy10', 'philosophy11', 'philosophy12',
                                'philosophy13', 'philosophy14', 'philosophy15', 'philosophy16',
                                'philosophy17', 'philosophy18', 'philosophy19', 'philosophy20',
                                'philosophy21', 'philosophy22', 'philosophy23', 'philosophy24',
                                'philosophy25', 'philosophy26', 'philosophy27', 'philosophy28',
                                'philosophy29', 'philosophy30', 'philosophy31', 'philosophy32',
                                'philosophy33', 'philosophy34', 'philosophy35']].astype(float)
                            
                return data, targets

            if subset == 'all_ordinal_scores':
                data = data[[   'age','gender_Female','gender_Male','gender_Other','gender_Prefer not to say',
                                'country_Argentina','country_Australia','country_Bangladesh','country_Canada',
                                'country_Chile','country_China - Hong Kong / Macau','country_Denmark','country_England/United Kingdom',
                                'country_Estonia','country_Finland','country_France','country_Germany','country_Ghana','country_Great Britain',
                                'country_Greece','country_Ireland','country_Israel and the Occupied Territories','country_Italy','country_Netherlands',
                                'country_New Zealand','country_Nigeria','country_Philippines','country_Poland','country_Portugal',
                                'country_Romania','country_Slovenia','country_South Africa','country_Spain','country_Sri Lanka',
                                'country_United States of America (USA)','country_Virgin Islands (UK)','country_Virgin Islands (US)',
                                'education_Bachelors Degree','education_High School','education_Masters Degree',
                                'education_Ph.D. or higher','education_Prefer not to say','education_Some High School',
                                'education_Trade School','extroversion_dim','agreeableness_dim','conscientiousness_dim',
                                'emotional_stability_dim','openness_dim','approval', 'love', 'achievement', 
                                'perfectionism','entitlement', 'omnipotence', 'autonomy']].astype(float)

                return data, targets


            elif subset == 'personality':
                data = data[['extroversion_dim', 'agreeableness_dim','conscientiousness_dim', 'emotional_stability_dim', 'openness_dim',]].astype(float)
                return data, targets
                          

            elif subset == 'values':
                data = data.loc[:, 'approval':'autonomy'].astype(float)   
                return data, targets

            elif subset == 'demo':
                data = data[['age','gender_Female','gender_Male','gender_Other','gender_Prefer not to say',
                             'country_Argentina','country_Australia','country_Bangladesh','country_Canada','country_Chile',
                             'country_China - Hong Kong / Macau','country_Denmark','country_England/United Kingdom',
                             'country_Estonia','country_Finland','country_France','country_Germany','country_Ghana','country_Great Britain',
                             'country_Greece','country_Ireland','country_Israel and the Occupied Territories','country_Italy',
                             'country_Netherlands','country_New Zealand','country_Nigeria','country_Philippines','country_Poland','country_Portugal',
                             'country_Romania','country_Slovenia','country_South Africa','country_Spain','country_Sri Lanka','country_United States of America (USA)',
                             'country_Virgin Islands (UK)','country_Virgin Islands (US)','education_Bachelors Degree','education_High School','education_Ph.D. or higher',
                             'education_Prefer not to say','education_Some High School','education_Trade School']].astype(float)

                return data, targets


            elif subset == 'all_+_values_binary':
                data = data.loc[:, 'age':'philosophy35'].astype(float)  

                data = pd.get_dummies(data, columns=[
                                                    'extraversion_enthusiastic', 'critical_quarrelsome','dependable_self-disciplined',
                                                    'anxious_easily_upset','openness_to_experiences', 'reserved_quiet', 'sympathetic_warm',
                                                    'disorganized_careless', 'calm_emotionally_stable','conventional_uncreative'
                                                    'philosophy1', 'philosophy2', 'philosophy3', 'philosophy4',
                                                    'philosophy5', 'philosophy6', 'philosophy7', 'philosophy8',
                                                    'philosophy9', 'philosophy10', 'philosophy11', 'philosophy12',
                                                    'philosophy13', 'philosophy14', 'philosophy15', 'philosophy16',
                                                    'philosophy17', 'philosophy18', 'philosophy19', 'philosophy20',
                                                    'philosophy21', 'philosophy22', 'philosophy23', 'philosophy24',
                                                    'philosophy25', 'philosophy26', 'philosophy27', 'philosophy28',
                                                    'philosophy29', 'philosophy30', 'philosophy31', 'philosophy32',
                                                    'philosophy33', 'philosophy34', 'philosophy35']) 

                return data, targets
                
            elif subset == 'summary_binary_set':

                    data = data[['age','gender_Female','gender_Male','gender_Other','gender_Prefer not to say',
                            'education_Bachelors Degree','education_High School','education_Ph.D. or higher',
                            'education_Prefer not to say','education_Some High School','education_Trade School',
                            'approval_level', 'love_level','achievement_level','perfectionism_level',
                            'entitlement_level','autonomy_level','omnipotence_level',
                            'extroversion_level', 'agreeableness_level','conscientiousness_level',
                             'emotional_stability_level', 'openness_level']]
                    return data, targets


if __name__ == "__main__" :
    main()
    # all_data = GetData()
    # data = all_data.get_data()
    # data = all_data.feature_process(data)
    # values = all_data.values(data,tipi1)
    # df= all_data.new_dataset(data,tipi1,values)
  
