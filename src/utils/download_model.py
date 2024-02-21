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