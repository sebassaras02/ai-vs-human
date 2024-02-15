from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from joblib import dump
import numpy as np
from sklearn.decomposition import PCA
import mlflow
from datetime import datetime

class FeatureTextExtraction:

    def __init__(self, mlflow_uri : str, mlflow_experiment_name : str, mlflow_run_name : str) -> None:
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.1, max_features=2000)
        self.pca = PCA(5, random_state=99)
        self.mlflow_uri = mlflow_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_run_name = mlflow_run_name
        # set the mlflow uri
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)
    
    def fit_tfidf(self, df: pd.DataFrame) -> None:
        """
        This function fits the model to the data
        
        Args:
            df: pd.DataFrame: The dataframe containing the data

        Returns:
            None
        """
        self.df = df
        self.df = self.df.dropna(subset=["processed_text"])
        self.matrix = self.vectorizer.fit_transform(df["processed_text"])
       
    def dimesion_reduction(self) -> pd.DataFrame:
        """
        This function reduces the dimension of the data

        Returns:
            pd.DataFrame: The dataframe containing the transformed data
        """
        self.reduced_data = self.pca.fit_transform(self.matrix.toarray())
        # convert to dataframe
        self.reduced_df = pd.DataFrame(self.reduced_data, columns=["dim1", "dim2", "dim3", "dim4", "dim5"])
        return self.reduced_df
    
    def fit_transform(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        This function fits the model to the data
        
        Args:
            df: pd.DataFrame: The dataframe containing the data

        Returns:
            pd.DataFrame: The dataframe containing the transformed data
        """
        with mlflow.start_run(run_name = self.mlflow_run_name + " " + datetime.today().strftime("%Y-%m-%d %H:%M:%S")):
            # log the parameters of the TF-IDF model
            self.fit_tfidf(df)
            # log the model of the TF-IDF model
            mlflow.sklearn.log_model(self.vectorizer, "tfidf_model")
            # log the parameters of the PCA model
            self.data = self.dimesion_reduction()
            # log the model of the PCA model
            mlflow.sklearn.log_model(self.pca, "pca_model")
            # end the run
            mlflow.end_run()
            # delete the parameters
            self.final_df = pd.concat([self.df, self.data], axis=1)
        return self.final_df
