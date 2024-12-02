from tensorflow import keras
from keras._tf_keras.keras.models import load_model
import numpy as np
from keras._tf_keras.keras.preprocessing import image


test_image_path = 'image_manual_testing/forest_image_test.jpeg'
model = load_model('models/resisc45_model_v3.keras')  # Replace with your saved model's path

img = image.load_img(test_image_path, target_size=(224, 224)) # Adjust target_size as needed
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize if needed

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
print(f'Predicted class: {predicted_class}')

