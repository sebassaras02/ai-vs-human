from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from joblib import dump
import numpy as np
from sklearn.decomposition import PCA


class FeatureTextExtraction:

    def __init__(self)->None:
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.1, max_features=2000)
        self.pca = PCA(5, random_state=99)
    
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
        self.fit_tfidf(df)
        self.data = self.dimesion_reduction()
        # save the tf-idf model 
        dump(self.vectorizer, "vectorizer_tfidf.joblib")
        # save the pca model
        dump(self.pca, "dim_reduction.joblib")
        return self.data
