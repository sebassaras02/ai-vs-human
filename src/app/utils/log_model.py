import mlflow
from datetime import datetime
from sklearn.metrics import classification_report

class LogModel:

    def __init__(self, mlflow_uri : str, mlflow_experiment_name : str, mlflow_run_name : str, X_train, Y_train, X_test, Y_test, model, model_name) -> None:
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
        self.report1 = classification_report(self.Y_test, self.model.predict(self.X_test), output_dict=True)
        mlflow.log_metric("accuracy", self.report1.pop("accuracy"))
        for class_or_avg, metrics_dict in self.report1.items():
            for metric, value in metrics_dict.items():
                mlflow.log_metric(class_or_avg + '_' + metric,value)

    def evaluate_test_data(self):
        """
        This function evaluates the model on the test data
        """
        self.report2 = classification_report(self.Y_test, self.model.predict(self.X_test), output_dict=True)
        mlflow.log_metric("accuracy", self.report2.pop("accuracy"))
        for class_or_avg, metrics_dict in self.report2.items():
            for metric, value in metrics_dict.items():
                mlflow.log_metric(class_or_avg + '_' + metric,value)
                
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
