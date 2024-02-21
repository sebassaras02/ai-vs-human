import numpy as np
import pandas as pd
import re
import mlflow
from dotenv import load_dotenv
from joblib import dump, load
import sys

sys.path.append('../utils')
from text_processing import TextProcessing


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