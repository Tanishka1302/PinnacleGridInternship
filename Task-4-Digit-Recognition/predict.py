import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# Load trained model
model = tf.keras.models.load_model("model/digit_model.keras")

# Load test dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Normalize images
x_test = x_test / 255.0

# Select image index
index = 10

# Predict
prediction = model.predict(x_test[index].reshape(1, 28, 28))
digit = np.argmax(prediction)

# Display image
plt.imshow(x_test[index], cmap="gray")
plt.title(f"Actual: {y_test[index]} | Predicted: {digit}")
plt.axis("off")

# Save screenshot
plt.savefig("screenshots/prediction.png")

plt.show()

print("Actual Digit :", y_test[index])
print("Predicted Digit :", digit)