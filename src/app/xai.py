import numpy as np
import pandas as pd
import sys
from lime.lime_text import LimeTextExplainer

sys.path.append('../')

from app.pipelines import pipeline_inference

def f(x):
    results = np.zeros((len(x), 2))  # Asumiendo que num_classes es la cantidad de clases en tu problema
    for i, element in enumerate(x):
        predictions = pipeline_inference(element)
        results[i, :] = predictions
    return results


def get_explanation(text):
    explainer = LimeTextExplainer(class_names=["Human", "AI"])
    explanation = explainer.explain_instance(
            text_instance = text,
            classifier_fn = f,
            num_features=30,
            num_samples = 10
        )
    a = explanation.as_list()
    result = {element[0]: element[1] for element in a}
    return result