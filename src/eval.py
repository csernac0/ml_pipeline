import pandas as pd
import pickle
import numpy as np
import sklearn.metrics as metrics

class EvalModel(object):
    """
    Evaluation class
    Created on April 21, 2021
    """
    def __init__(self, file_name: str = 'df_processed_test.csv', single_obs: pd.DataFrame = pd.DataFrame()):
        print('\nInitializing EvalModel class ...')
        with open('obj/model.p', 'rb') as handle:
            self.model = pickle.load(handle)

        with open('src/features_train.p', 'rb') as handle:
            features_dict = pickle.load(handle)
        #get the dict in separate vars
        self.features = features_dict['features']
        self.obj_var = features_dict['obj_var']
        if single_obs.empty:
            self.individual_pred = 0
            self.df = pd.read_csv('data/'+file_name)
        else:
            self.individual_pred = 1
            self.df = single_obs
    
    def measure_results(self, y_true, y_pred):
        # Regression metrics
        mae = np.round(metrics.mean_absolute_error(y_true, y_pred),3)
        mse = np.round(metrics.mean_squared_error(y_true, y_pred),3) 
        if self.individual_pred == 0:
            r2 = np.round(metrics.r2_score(y_true, y_pred),3)   
            print('     R2:',r2)
        r2 = 0
        scores = pd.DataFrame([['mae',mae], 
            ['mse',mse], 
            ['r2',r2]
            ],
            columns=['metric','value'])
        print('     metrics saved on obj/model/test_metrics.csv')
        scores.to_csv('obj/test_metrics.csv',index=False)
        

    def eval_model(self):
        """"""
        predictions = self.model.predict(self.df[self.features])
        self.measure_results(self.df[self.obj_var], predictions)
        if self.individual_pred == 1:
            print('\npredicted price:',predictions[0])
        
