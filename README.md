# 🧑‍🔬 ML Pipeline

ml_pipeline is a tool that allows data scientists to easily run ML pipelines on kaggle competition datasets. 
For default it will run on https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview competiton.


## Prerequisites 
1. python 3.8
2. Kaggle **API** credentials: To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (https://www.kaggle.com/username/account) and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. 
  For further documentation please visit: https://github.com/Kaggle/kaggle-api
3. Generate a **.env** file with the generated kaggle.json values. This file must be saved at this level with the following structure.
    ```bash
    kaggle_username=your_username
    kaggle_key=your_key   
    ```
4. Accept the competition rules so we can download files throughout Kaggle API.



## Instalation
After cloning this repo, please install requirements.
```python
pip install -r requirements.txt
```


## Usage
To run the entire pipeline (extraction -> data_processing -> model_training -> model eval) please run. 
```python
python index_ml_pipeline.py
```

And for a single evaluation (must run first the training pipeline).
```python
python individual_pred_pipeline.py id_to_predict
```
For house id 7
```python
python individual_pred_pipeline.py 7
```


## Knowledge
It can be found on the [wiki](https://github.com/csernac0/ml_pipeline/wiki) of this project.
