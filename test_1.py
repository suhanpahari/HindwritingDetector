import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image

# Load the saved model
model = load_model('ML_1.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert('L')  # Convert image to grayscale
    image = image.resize((28, 28))  # Resize image to 28x28 pixels
    image = np.array(image)  # Convert image to numpy array
    image = image.reshape((1, 784))  # Flatten image into a single row
    image = image.astype('float32') / 255  # Preprocess image data
    return image

# Function to predict the digit from an image
def predict_digit(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_label = np.argmax(prediction)
    return predicted_label

# Example usage
image_path = 'test.png'  # Replace with the path to your image
image = Image.open(image_path)
predicted_digit = predict_digit(image)

# Display the image and predicted digit
plt.imshow(image, cmap='gray')
plt.title(f'Predicted Digit: {predicted_digit}')
plt.axis('off')
plt.show()
