import tensorflow as tf

# Load your existing model
model = tf.keras.models.load_model('model/pneumonia_model.h5')

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open('model/pneumonia_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Conversion complete! Saved as pneumonia_model.tflite")