from src.extraction import GetRawData
from src.processing import PrepareRawData
from src.train import TrainModel
from src.eval import EvalModel
import pickle


if __name__ == '__main__':
    #download raw data from kaggle competition
    GetRawData().get_kaggle_dataset(
        competition = 'house-prices-advanced-regression-techniques',
        file_name = 'train.csv'
    )

    #Save the desired features on pickle file
    features_dict = {
    'features': [
     'MSSubClass',
     'MSZoning',
     'LotFrontage',
     'LotArea',
     'Street',
     'Alley',
    ],
     'obj_var': 'SalePrice'}
    with open('src/features.p', 'wb') as handle:
        pickle.dump(features_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #PIPELINE STARTS
    #Instanciate PrepareRawData class
    obj_prepare = PrepareRawData()
    #get raw data
    obj_prepare.get_raw(file_name = 'train.csv')

    #Process 
    obj_prepare.get_numeric_categoric_vars()
    obj_prepare.process_categoric()
    #imput strategy can be one of the following
    #['most_frequent','mean','constant'->0]
    obj_prepare.impute_numerical(impute_strategy = 'constant')

    #default split_size = .3
    #obj_prepare.split_rnd_set()
    obj_prepare.split_rnd_set(split_size = .1)

    #model_type can be on of the following ['reg', 'gbr']
    TrainModel(file_name = 'df_processed_train.csv').train_main(model_type = 'gbr')

    EvalModel(file_name = 'df_processed_test.csv').eval_model()