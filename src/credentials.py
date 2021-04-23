#credentials.py
import os
import json
from dotenv import load_dotenv, find_dotenv

def save_kaggle_config_file():
    """
    Generates a configuration file kaggle.json
    Created on April 21, 2021

    Input
    ----------
        file : .env
            kaggle_username=your_user
            kaggle_key=your_key

    Output
    ------- 
        file : .json
            Places the file in $HOME/.kaggle/kaggle.json

    For further documentation please visit 
        - https://github.com/Kaggle/kaggle-api
        - https://pypi.org/project/python-dotenv/

    """

    try:
        #set the key values from .env file
        load_dotenv(find_dotenv())
        
        #get the envvars
        str_username = os.environ.get("kaggle_username")
        str_key = os.environ.get("kaggle_key")
        str_home = os.environ['HOME']
        
        #set the vars in the required format
        config = {"username":str_username,"key":str_key}
        #save the file
        with open(str_home + '/.kaggle/kaggle.json', 'w') as f:
            json.dump(config, f)
    except:
        print('please check the required arguments')

