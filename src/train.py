import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

class TrainModel(object):
    """
        Training class
        Created on April 21, 2021
    """
    def __init__(self, file_name: str = 'df_processed_train.csv'):
        """
        Get the train dataset saved in the previous step
        Parameters
        ----------
            file_name : string
                dataset to process (default: data/train.csv)
        """
        #read the features from config file_name
        print('\nInitializing TrainModel class...')
        with open('src/features_train.p', 'rb') as handle:
            features_dict = pickle.load(handle)

        #get the dict in separate vars
        self.features = features_dict['features']
        self.obj_var = features_dict['obj_var']

        #read dataset to process
        print('     reading data from','data/'+file_name)
        self.X_train = pd.read_csv('data/'+file_name)
        #save the objetive var independent
        self.y_train = self.X_train[self.obj_var]

        print('     Training on:', self.X_train.shape)

    def train_gbr(self):
        reg = GradientBoostingRegressor(random_state=0)
        reg.fit(self.X_train[self.features], self.y_train)
        return reg

    def train_reg(self):
        reg = LinearRegression()
        reg.fit(self.X_train[self.features], self.y_train)
        return reg

    def measure_results(self, y_true, y_pred):
        # Regression metrics
        mae = metrics.mean_absolute_error(y_true, y_pred) 
        mse = metrics.mean_squared_error(y_true, y_pred) 
        r2 = metrics.r2_score(y_true, y_pred)

        scores = pd.DataFrame([['mae',mae], 
            ['mse',mse], 
            ['r2',r2]],
            columns=['metric','value'])
        scores.to_csv('obj/train_metrics.csv',index=False)
        print('     R2:',r2)

    def train_main(self, model_type: str = 'gbr'):
        if model_type == 'gbr':
            self.model = self.train_gbr()
        elif model_type == 'reg':
            self.model = self.train_reg()
        self.y_pred = self.model.predict(self.X_train[self.features])
        self.measure_results(self.y_train, self.y_pred)
        print('     metric and model saved on obj/model/train_metrics.csv')
        with open('obj/model.p', 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

