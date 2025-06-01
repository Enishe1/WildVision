from src.train import train_and_evaluate
from src.evaluate import evaluate_model
from src.utils import create_directory_structure

if __name__ == "__main__":
    # Create directory structure
    create_directory_structure()
    
    print("Starting model training...")
    history = train_and_evaluate()
    print("Training completed!")
    
    print("\nEvaluating best model...")
    evaluate_model('models/best_model.h5')
    print("Evaluation completed!")
