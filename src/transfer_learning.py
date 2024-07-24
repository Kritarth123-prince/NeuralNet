import tensorflow as tf

def use_pretrained_model(base_model_name='VGG16', input_shape=(224, 224, 3), num_classes=10):
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

def fine_tune_model(model, base_model_name='VGG16', fine_tune_at=100):
    base_model = getattr(tf.keras.applications, base_model_name)(
        weights='imagenet', include_top=False)

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    return model
