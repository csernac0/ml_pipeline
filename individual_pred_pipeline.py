import sys
from src.extraction import GetRawData
from src.processing import PrepareRawData
from src.train import TrainModel
from src.eval import EvalModel

if __name__ == '__main__':
	#PIPELINE STARTS
    #Instanciate PrepareRawData class
    obj_prepare = PrepareRawData()
    #read downloaded raw data
    obj_prepare.get_raw(file_name = 'train.csv', raw_id = int(sys.argv[1]))
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