# 🧑‍🔬 ML Pipeline

ml_pipeline is a tool that allows data scientists to easily run ML pipelines on kaggle competition datasets.


## Prerequisites
1. python 3.8
2. Kaggle **API** credentials: To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. 
  For further documentation please visit: **https://github.com/Kaggle/kaggle-api**
3. Generate a **.env** file with the previous kaggle.json file. This file must be saved at this level.
      ```bash
          kaggle_username=your_username
          kaggle_key=your_key   
      ```
4. Accept the competition rules so we can use the API to download files.



## Instalation
After clone the repo, please make sure to do the following:
```python
pip install -r requirements.txt
```


## Usage
```python
python ml_pipeline.py
```

And for a single evaluation (must run first the training pipeline).
```python
python ml_pipeline.py id_to_predict
```
