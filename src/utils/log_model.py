import mlflow
from datetime import datetime
from sklearn.metrics import classification_report

class FeatureTextExtraction:

    def __init__(self, mlflow_uri : str, mlflow_experiment_name : str, mlflow_run_name : str, X_train, Y_train, X_test, Y_test, model, model_name) -> None:
        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.1, max_features=2000)
        self.pca = PCA(5, random_state=99)
        self.mlflow_uri = mlflow_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_run_name = mlflow_run_name
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.model_name = model_name
        self.model = model
        # set the mlflow uri
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)
    
    def evaluate_train_data(self):
        """
        This function evaluates the model on the training data
        """
        self.report1 = classification_report(self.Y_train, self.model.predict(self.X_train))
        for label, metrics in self.report1.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (float, int)):  
                    mlflow.log_metric(f"{label}_{metric_name}", value)
    def evaluate_test_data(self):
        """
        This function evaluates the model on the test data
        """
        self.report2 = classification_report(self.Y_test, self.model.predict(self.X_test))
        for label, metrics in self.report2.items():
            for metric_name, value in metrics.items():
                if isinstance(value, (float, int)):
                    mlflow.log_metric(f"{label}_{metric_name}", value)

    def register_model(self):
        """
        This function register the model created parameters and the model
        """
        params = self.model.get_params()
        mlflow.log_params(params)
        mlflow.sklearn.log_model(self.model, self.model_name)

    def fit_transform(self):
        with mlflow.start_run(run_name = self.mlflow_run_name + " " + datetime.today().strftime("%Y-%m-%d %H:%M:%S")):
            self.evaluate_train_data()
            self.evaluate_test_data()
            self.register_model()
            mlflow.end_run()
        print("Model performance over the test dataset")
        print(self.report2)
