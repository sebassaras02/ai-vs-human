import numpy as np
import pandas as pd
import re
import mlflow
from dotenv import load_dotenv
from joblib import dump, load
import sys

sys.path.append('../utils')
from text_processing import TextProcessing

def pipeline_download_models():
    """
    This function downloads the models from the mlflow server and saves them in the models folder

    Args:
        None

    Returns:
        None
    """
    load_dotenv('../../.env')
    # download the tf-idf model
    tfidf_logged_model = 'runs:/a63128b897bd4f91a06f20939a715b98/tfidf_model'
    tfidf_model = mlflow.sklearn.load_model(tfidf_logged_model)
    dump(tfidf_model, '../../models/tfidf_model.joblib')
    # download the pca model
    pca_logged_model = 'runs:/a63128b897bd4f91a06f20939a715b98/pca_model'
    pca_model = mlflow.sklearn.load_model(pca_logged_model)
    dump(pca_model, '../../models/pca_model.joblib')
    # download the classifier
    classifier_logged_model = 'runs:/49483b7a0f95430a8492a448ac13e8d7/random-forest'
    classifier_model = mlflow.sklearn.load_model(classifier_logged_model)
    dump(classifier_model, '../../models/classifier_model.joblib')


def pipeline_inference(input : str):
    # load tf-idf model
    tfidf_model = load('../../models/tfidf_model.joblib')
    # load pca model
    pca_model = load('../../models/pca_model.joblib')
    # load the model
    classifier_model = load('../../models/classifier_model.joblib')

    # preprocess the input
    text_processing = TextProcessing()
    text_processed = text_processing.fit_transform_text(input)
    vector = tfidf_model.transform([text_processed])
    vector_pca = pca_model.transform(vector)
    # make a vector with the pca values
    df = pd.DataFrame(vector_pca, columns = ["dim1", "dim2", "dim3", "dim4", "dim5"])
    # make the prediction
    prediction = classifier_model.predict_proba(df)
    return prediction