import sys
from src.extraction import GetRawData
from src.processing import PrepareRawData
from src.train import TrainModel
from src.eval import EvalModel

if __name__ == '__main__':
    obj_prepare = PrepareRawData()
    obj_prepare.get_raw(file_name = 'train.csv', raw_id = int(sys.argv[1]))
    obj_prepare.get_numeric_categoric_vars()
    obj_prepare.process_categoric(raw_id = True)
    obj_prepare.impute_numerical(raw_id = True)

    obs_to_eval = obj_prepare.get_single_obs()
    obj_eval = EvalModel(single_obs = obs_to_eval)
    obj_eval.eval_model()