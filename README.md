# 🧑‍🔬 ML Training Pipeline

ml_pipeline is a tool that allows data scientists to easily run ML training pipelines on kaggle competition datasets. 
The pipeline includes: 

* Extraction (from kaggle competition)
* Data Processing (Treatment for nan, OneHotEncoding, split dataset)
* Model Training 
* Model Eval

## Prerequisites 
1. python 3.8
2. Kaggle **API** credentials: To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (https://www.kaggle.com/username/account) and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. 
  For further documentation please visit: https://github.com/Kaggle/kaggle-api
3. Please generate a **.env** file with the generated kaggle.json values from prevous step. This file must be saved at this level with the following structure.
    ```bash
    kaggle_username=your_username
    kaggle_key=your_key   
    ```
4. **It is very important** to accept the competition rules so we can download files throughout Kaggle API.
5. There is a competition_config.json file included, feel free to change the values for another competition dataset. The current file is configured for https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview 

    ```bash
    {
        "competition_name" : "house-prices-advanced-regression-techniques",
        "file_download": "train.csv",
        "features" : [
            "MSSubClass",
            "MSZoning",
            "LotFrontage",
            "LotArea",
            "Street",
            "Alley"
        ],
        "obj_var" : "SalePrice",
        "id_dataset" : "Id",
        "impute_numeric" : "most_frequent", # Can be: most_frequent, mean, constant = 0
        "model" : "gbr" # Can be gbr= GradientBoostingRegresor, reg= LinearRegression
    }  
    ```
    
    For example, for another kaggle dataset https://www.kaggle.com/anmolkumar/house-price-prediction-challenge?select=train.csv: 

      ```bash
        {
            "type_kaggle_object" : "dataset",
            "competition_name" : "anmolkumar/house-price-prediction-challenge",
            "file_download": "train.csv",
            "features" : [
                "POSTED_BY", 
                "UNDER_CONSTRUCTION", 
                "RERA", 
                "BHK_NO.", 
                "BHK_OR_RK",
                "SQUARE_FT", 
                "READY_TO_MOVE", 
                "RESALE", 
                "ADDRESS", 
                "LONGITUDE",
                "LATITUDE"
            ],
            "obj_var" : "TARGET(PRICE_IN_LACS)",
            "id_dataset" : "index",
            "impute_numeric" : "most_frequent",
            "model" : "gbr"
        }  
      ```

## Installation
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
Example: for row `id = 7`
```python
python individual_pred_pipeline.py 7
```


## Knowledge
For more information about this project please visit the [wiki](https://github.com/csernac0/ml_pipeline/wiki) of this project.
It includes a description for each folder and file.
