from src.extraction import GetRawData
from src.processing import PrepareRawData
from src.train import TrainModel
from src.eval import EvalModel
import json

def get_config_vars():
    """
    Reads competition_config.json
    Output
    ------
        competition_name: str,
        file_dowload: str,
            file to train pipeline
        imput_numeric: str,
            how we are going to impute nan numeric features
        model: str
            gbr, reg
    """
    #Read config file
    with open('competition_config.json') as json_file:
        competition_config = json.load(json_file)

    type_kaggle_object = competition_config['type_kaggle_object']
    competition_name = competition_config['competition_name']
    file_download = competition_config['file_download']
    features = competition_config['features']
    obj_var = competition_config['obj_var']
    id_dataset = competition_config['id_dataset']
    impute_numeric = competition_config['impute_numeric']
    model = competition_config['model']

    return [
        type_kaggle_object,
        competition_name, 
        file_download, 
        features, 
        obj_var, 
        id_dataset, 
        impute_numeric, 
        model
    ]


if __name__ == '__main__':
    #PIPELINE STARTS
    

    #1. Get vars from config file
    (
        type_kaggle_object,
        competition_name, 
        file_download, 
        features, 
        obj_var, 
        id_dataset, 
        impute_numeric, 
        model
    ) = get_config_vars()
    
    #2. download raw data from kaggle competition
    GetRawData(type_kaggle_object).get_kaggle_dataset(
        competition = competition_name,
        file_name = file_download
    )
    
    #3. Instanciate PrepareRawData class
    obj_prepare = PrepareRawData(        
        features = features, 
        obj_var = obj_var, 
        id_dataset = id_dataset, 
        impute_numeric = impute_numeric
        )
    #read downloaded raw data
    obj_prepare.get_raw(
        file_name = file_download
    )
    #Classify numerical & categorical features  
    obj_prepare.get_numeric_categoric_vars()
    #Processing categorical features  
    obj_prepare.process_categoric()
    #Processing numerical nan imputation
    obj_prepare.impute_numerical()
    #Split data into train & test
    obj_prepare.split_rnd_set(split_size = .1)

    #4. Train model
        #model_type can be on of the following ['reg', 'gbr']
    TrainModel(
        model = model
    ).train_main()
    #5. Eval model
        #please visit obj/ to see results
    EvalModel().eval_model()

