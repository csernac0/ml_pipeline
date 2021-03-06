import sys
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

    competition_name = competition_config['competition_name']
    file_download = competition_config['file_download']
    features = competition_config['features']
    obj_var = competition_config['obj_var']
    id_dataset = competition_config['id_dataset']
    impute_numeric = competition_config['impute_numeric']
    model = competition_config['model']

    return [
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
        competition_name, 
        file_download, 
        features, 
        obj_var, 
        id_dataset, 
        impute_numeric, 
        model
    ) = get_config_vars()

    #Instanciate PrepareRawData class
    obj_prepare = PrepareRawData(        
        features = features, 
        obj_var = obj_var, 
        id_dataset = id_dataset, 
        impute_numeric = impute_numeric
    )
    #read downloaded raw data
    obj_prepare.get_raw(
        file_name = file_download, 
        raw_id = int(sys.argv[1])
        )
    #Classify numerical & categorical features  
    obj_prepare.get_numeric_categoric_vars()
    #Processing categorical features  
    obj_prepare.process_categoric(raw_id = True)
    #Processing numerical nan imputation
    obj_prepare.impute_numerical(raw_id = True)

    #Eval model
    obs_to_eval = obj_prepare.get_single_obs()
    obj_eval = EvalModel(single_obs = obs_to_eval)
    obj_eval.eval_model()

