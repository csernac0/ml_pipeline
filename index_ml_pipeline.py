from src.extraction import GetRawData
from src.processing import PrepareRawData
from src.train import TrainModel
from src.eval import EvalModel
import pickle


def select_features(
    list_features: list = ['MSSubClass'], 
    obj_var: str = 'SalePrice'
):
    """
    Save the selected features on pickle file
    Parameters
    ----------
        list_features : list,
        obj_var : str
    """
    features_dict = {
    'features': list_features,
     'obj_var': obj_var
     }
    #Saving features in a config pickle
    with open('src/features.p', 'wb') as handle:
        pickle.dump(
            features_dict, 
            handle, 
            protocol=pickle.HIGHEST_PROTOCOL
        )

if __name__ == '__main__':
    #download raw data from kaggle competition
    GetRawData().get_kaggle_dataset(
        competition = 'house-prices-advanced-regression-techniques',
        file_name = 'train.csv'
    )

    #Select the features
    select_features(
        [
            'MSSubClass',
            'MSZoning',
            'LotFrontage',
            'LotArea',
            'Street',
            'Alley'
        ]
        ,'SalePrice'
    )

    #PIPELINE STARTS
    #Instanciate PrepareRawData class
    obj_prepare = PrepareRawData()
    #read downloaded raw data
    obj_prepare.get_raw(file_name = 'train.csv')
    #Classify numerical & categorical features  
    obj_prepare.get_numeric_categoric_vars()
    #Processing categorical features  
    obj_prepare.process_categoric()
    #Processing numerical nan imputation
        #imput strategy can be one of the following
        #['most_frequent','mean','constant'->0]
    obj_prepare.impute_numerical(impute_strategy = 'constant')
    #Split data into train & test
        #default split_size = .3
        #obj_prepare.split_rnd_set()
    obj_prepare.split_rnd_set(split_size = .1)

    #Train model
        #model_type can be on of the following ['reg', 'gbr']
    TrainModel(
        file_name = 'df_processed_train.csv'
    ).train_main(model_type = 'gbr')
    #Eval model
        #please visit obj/ to see results
    EvalModel(file_name = 'df_processed_test.csv').eval_model()