from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .utils import download_species_images
import config
import json

def prepare_data():
    """Prepare data generators"""
    # Download images if not already exists
    if not os.path.exists("data/species_metadata.csv"):
        download_species_images(
            config.TARGET_SPECIES,
            max_images=config.MODEL_CONFIG["max_images_per_species"]
        )
    
    # Create data augmentation pipeline
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=config.MODEL_CONFIG["test_size"],
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=config.MODEL_CONFIG["test_size"]
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        'data/images',
        target_size=config.MODEL_CONFIG["image_size"],
        batch_size=config.MODEL_CONFIG["batch_size"],
        class_mode='categorical',
        subset='training',
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        'data/images',
        target_size=config.MODEL_CONFIG["image_size"],
        batch_size=config.MODEL_CONFIG["batch_size"],
        class_mode='categorical',
        subset='validation',
        seed=42
    )
    
    # Save class indices
    with open('models/class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
    
    return train_generator, val_generator
