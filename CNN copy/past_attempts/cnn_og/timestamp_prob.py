import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np

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

# Predict the probabilities for the test data
predictions = model.predict(X_test_reshaped)

# Output probabilities for each timestamp
probabilities_df = pd.DataFrame(predictions, columns=['nothing_prob', 'throw_prob', 'swing_prob'])
print(probabilities_df)

# If you want to save the probabilities to a CSV file
probabilities_df.to_csv('output_probabilities.csv', index=False)

# Convert probabilities to predicted labels (0 for nothing, 1 for throw, 2 for swing) with custom threshold
predicted_labels = []
for prob in predictions:
    nothing_prob, throw_prob, swing_prob = prob
    if nothing_prob < 0.85:
        # Choose between throw or swing based on which has the higher probability
        if throw_prob > swing_prob:
            predicted_labels.append(1)  # Throw
        else:
            predicted_labels.append(2)  # Swing
    else:
        predicted_labels.append(0)  # Nothing

# Convert list to array for easier counting
predicted_labels = np.array(predicted_labels)

# Count the number of swings (2) and throws (1)
num_swings = (predicted_labels == 2).sum()
num_throws = (predicted_labels == 1).sum()

print(f"Number of swings: {num_swings}")
print(f"Number of throws: {num_throws}")

