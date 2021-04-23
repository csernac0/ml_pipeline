import os
from os import path
import pandas as pd
from src.credentials import save_kaggle_config_file

class GetRawData(object):
    """
    Gets a desired raw dataset    
    Created on April 21, 2021
    """
    def __init__(self, type_kaggle_object: str = 'competition'):
        """Check if the credentials are properly configured"""
        print('\nInitializing GetRawData class...')
        str_home = os.environ['HOME']
        self.type_kaggle_object = type_kaggle_object
        #save the file
        if not path.exists(str_home+'/.kaggle/kaggle.json'):
            save_kaggle_config_file()
            

    def get_kaggle_dataset(
        self, 
        competition: str, 
        file_name: str
    ):
        from kaggle.api.kaggle_api_extended import KaggleApi
        """
        Gets a kaggle competition dataset
        Mandatory to accept competition rules with used credentials
        
        Parameters
        ----------
            competition : string
                Competition kaggle name
                example: 'house-prices-advanced-regression-techniques'

            file_name : string 
                name of the file for the given competition.

        Output
        ------- 
            file : .csv
                Places the file in data/file_name

        For further documentation please visit 
            - https://github.com/Kaggle/kaggle-api
        """
        try:
            #Instanciates kagle api class
            api = KaggleApi()
            #Read credentials
            api.authenticate()
            #download dataset

            if self.type_kaggle_object == 'competition':
                api.competition_download_file(
                    competition, 
                    file_name
                )
            else:
                api.dataset_download_files(
                    competition, 
                    unzip=True
                )
            #Save it in data folder
            os.rename(
                file_name, 
                'data/'+file_name
            )
        except:
            print('\tplease check the required vars')


    def get_local_dataset(
        self, 
        file_name: str
    ):
        """Function for illustration     purposes"""
        pd.read_csv(file_name)
        #save

    