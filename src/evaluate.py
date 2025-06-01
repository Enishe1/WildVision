import tensorflow as tf
import numpy as np
import json
from sklearn.metrics import classification_report
import pandas as pd
from .data_loader import prepare_data

def evaluate_model(model_path='models/best_model.h5'):
    """Evaluate a trained model"""
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Prepare data
    _, val_gen = prepare_data()
    
    # Load class indices
    with open('models/class_indices.json') as f:
        class_indices = json.load(f)
    class_names = list(class_indices.keys())
    
    # Evaluate
    results = model.evaluate(val_gen)
    print(f"Evaluation results: {results}")
    
    # Generate predictions
    true_labels = val_gen.classes
    predictions = model.predict(val_gen)
    pred_labels = np.argmax(predictions, axis=1)
    
    # Classification report
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=class_names,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('reports/model_evaluation_report.csv')
    print("Evaluation report saved to reports/model_evaluation_report.csv")
    
    return report_df
