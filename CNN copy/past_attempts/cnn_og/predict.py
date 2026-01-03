import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = tf.keras.models.load_model("sports_tracker_model.h5")

# Load the test data (without the label column)
file_path = 'test_data_Nick_300-400_throws.csv'  # Update this with the path to your test data file
test_data = pd.read_csv(file_path)

# Prepare the test dataset
# Remove the 'time_trame' column
X_test = test_data.drop(columns=['time_trame'])

# Normalize the test sensor data using the same scaler as before
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Reshape the test data to match the input shape (samples, 1, features)
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Predict the labels for the test data
predictions = model.predict(X_test_reshaped)

# Convert predictions from one-hot encoded format back to class labels (0, 1, 2)
predicted_labels = predictions.argmax(axis=1)

# Count the number of swings (2) and throws (1)
num_swings = (predicted_labels == 2).sum()
num_throws = (predicted_labels == 1).sum()

print(f"Number of swings: {num_swings}")
print(f"Number of throws: {num_throws}")

