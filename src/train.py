import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os
from sklearn.metrics import confusion_matrix, classification_report
from .model import create_model
from .data_loader import prepare_data
from .utils import create_directory_structure
import config
import pandas as pd

def train_and_evaluate():
    """Main training and evaluation pipeline"""
    # Setup environment
    create_directory_structure()
    
    # Prepare data
    train_gen, val_gen = prepare_data()
    num_classes = len(train_gen.class_indices)
    
    # Create model
    model = create_model(num_classes)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.h5',
            save_best_only=True,
            monitor='val_top5_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.CSVLogger('models/training_log.csv')
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        epochs=config.MODEL_CONFIG["epochs"],
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    # Save final model
    model.save('models/final_model.h5')
    
    # Generate evaluation plots
    generate_evaluation_plots(history, model, val_gen)
    
    return history

def generate_evaluation_plots(history, model, val_gen):
    """Generate evaluation visualizations"""
    # Training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png')
    
    # Confusion matrix
    true_labels = val_gen.classes
    predictions = model.predict(val_gen)
    pred_labels = np.argmax(predictions, axis=1)
    
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    
    # Classification report
    with open('models/class_indices.json') as f:
        class_indices = json.load(f)
    class_names = list(class_indices.keys())
    
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=class_names,
        output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv('reports/classification_report.csv')
    
    # Per-class accuracy
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_mask = (true_labels == i)
        if sum(class_mask) > 0:  # Avoid division by zero
            class_acc = np.mean(pred_labels[class_mask] == i)
            class_accuracy[class_name] = class_acc
    
    plt.figure(figsize=(10, 25))
    sns.barplot(
        x=list(class_accuracy.values()),
        y=list(class_accuracy.keys()),
        orient='h'
    )
    plt.title('Per-Class Accuracy')
    plt.xlabel('Accuracy')
    plt.tight_layout()
    plt.savefig('plots/per_class_accuracy.png')
    plt.close()
