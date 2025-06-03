from tensorflow.keras import layers, models, applications
import tensorflow as tf
import config

def create_model(num_classes):
    """Create and compile the model"""
    # Base model selection
    base_model_class = getattr(applications, config.MODEL_CONFIG["base_model"])
    
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_shape=(*config.MODEL_CONFIG["image_size"], 3)
    )
    base_model.trainable = False
    
    # Build model
    inputs = tf.keras.Input(shape=(*config.MODEL_CONFIG["image_size"], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
        ]
    )
    
    return model
