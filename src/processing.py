import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

class PrepareRawData(object):
    """
        Basic cleaning and processing raw data
        Created on April 21, 2021
    """
    def __init__(
        self, 
        features: list, 
        obj_var: str, 
        id_dataset: str, 
        impute_numeric: str
    ):
        """
        Set config vars
        """
        print('\nInitializing PrepareRawData class...')

        #get the dictionary in separate vars
        self.features = features
        self.obj_var = obj_var
        self.id_dataset = id_dataset
        self.impute_numeric = impute_numeric


    def get_raw(
        self, 
        file_name: str = '', 
        raw_id: int = np.nan
    ):
        """
        Get raw data, 
        if raw_id is sent then it will filter the data
        This is for the individual prediction pipeline
        Optional Parameters
        ----------
            [file_name]: string, 
            [raw_id]: int
        """
        if file_name != '':
            #read dataset to process
            print(
                '\treading data from',
                'data/'+file_name
            )
            try:
                self.df = pd.read_csv('data/'+file_name)
                if not pd.isnull(raw_id):
                    print('\tgetting raw id:', raw_id)
                    try:
                        #Nota hay que dejar el id configurable
                        self.df = self.df[
                            self.df[self.id_dataset] == raw_id
                        ]
                    except:
                        print("\tdoesn't exist id in dataset")
            except:
                print('\terror on reading file')
            finally:
                #save the objetive var independent
                self.df_y = self.df[self.obj_var] 
                self.df = self.df[self.features]


    def get_numeric_categoric_vars(self):
        """
        Function that identifies 
        categorical and numeric features
        based on df.dtypes
        """
        self.numeric_cols = []
        self.cat_cols = []
        #identify type for each col in features 
        for col in self.features:
            if self.df[col].dtype == 'object':
                #save cat cols in array
                self.cat_cols.append(col)
            else:
                #save numeric cols in array
                self.numeric_cols.append(col)


    def process_categoric(
        self, 
        raw_id: bool = np.nan
    ):
        """
        Fill nan with 'other' category 
        only for categorical features
        Parameters
        ----------
            raw_id: bool
            flow for individual predictions
        Create dummies
        """
        print(
            '\tprocessing categorical features ...'
        )
        #fill na with 'other' value
        self.df[self.cat_cols] = self.df[
            self.cat_cols
        ].fillna('other')
        
        #if not single eval it must rtrain an encoder       
        if pd.isnull(raw_id):
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(self.df[self.cat_cols])
            #save encoder
            with open('obj/encode_categorical.p', 'wb') as handle:
                pickle.dump(
                    enc, 
                    handle, 
                    protocol=pickle.HIGHEST_PROTOCOL
                )
        else:
            #if is single eval it must read encoder previously trained
            try:
                print('\tread saved encoder')
                with open('obj/encode_categorical.p', 'rb') as handle:
                    enc = pickle.load(handle)
            except:
                print('\tmust exist a categorical encoder')

        #save dummies
        self.df_cat = pd.DataFrame(
            enc.transform(self.df[self.cat_cols]).toarray(),
            columns = enc.get_feature_names(self.cat_cols)
        )


    def impute_numerical(
        self, 
        raw_id: bool = np.nan
    ):
        """
        Imputation transformer for completing missing values.
        Parameters
        ----------
            impute_strategy: string, 
                'mean', 'median', 'most_frequent', 'constant'(default)
            raw_id: bool
                flow for individual predictions
        """
        print('\tprocessing numerical features ...')
        #if not single eval it must train an imputer     
        if pd.isnull(raw_id):
            imputer = SimpleImputer(
                missing_values=np.nan, 
                strategy=self.impute_numeric
            )
            imputer.fit(self.df[self.numeric_cols])
            #save imputer
            with open('obj/impute_numerical.p', 'wb') as handle:
                pickle.dump(
                    imputer, 
                    handle, 
                    protocol=pickle.HIGHEST_PROTOCOL
                )
        else:
            #if it is single eval it must read imputer previously trained
            try:
                print('\tread saved imputer')
                with open('obj/impute_numerical.p', 'rb') as handle:
                    imputer = pickle.load(handle)
            except:
                print('\tmust exist an imputer')

        #save the new imputed values
        self.df = pd.DataFrame(
            imputer.transform(self.df[self.numeric_cols]), 
            columns = self.numeric_cols
        )


    def split_rnd_set(
        self, 
        split_size: float = .3
    ):
        """
        Rnd train test split
        Parameters
        ----------
        size : float
            proportion for train & test
        """
        #concat numeric & dummies features
        self.df = pd.concat([self.df, self.df_cat], axis=1)
        #rnd split
        df_train, df_test, df_train_y, df_test_y = train_test_split(
            self.df, 
            self.df_y, 
            test_size=split_size, 
            random_state=42
        )
        #getting the new features (after encoding)
        features_dict = {
            'features': df_train.columns,
             'obj_var': 'SalePrice'
        }

        #rsaving files
        df_train = pd.concat([df_train, df_train_y], axis=1)
        df_test = pd.concat([df_test, df_test_y], axis=1)

        df_train.to_csv(
            'data/df_processed_train.csv', 
            index=False
        )
        df_test.to_csv(
            'data/df_processed_test.csv', 
            index=False
        )
        #save the features for training
        with open('src/features_train.p', 'wb') as handle:
            pickle.dump(
                features_dict, 
                handle, 
                protocol=pickle.HIGHEST_PROTOCOL
            )

        print('\n\tSplit done')
        print(
            '\t\tdata/df_processed_train.csv shape:', 
            df_train.shape
        )
        print(
            '\t\tdata/df_processed_test.csv shape:', 
            df_test.shape
        )


    def get_single_obs(self):
        """
        Only for single eval pipeline
        Return the dataframe with single observation
        """
        #concat numeric & dummies features
        self.df_y = self.df_y.reset_index()
        del self.df_y['index']
        return pd.concat(
            [
                self.df, 
                self.df_cat, 
                self.df_y
            ], axis=1
        )
        

    def split_stratified_set(self):
        """function created for illustration purposes"""
        #For example if we want to create a split time based
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.33, 
            random_state=42
        )


