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

# Output probabilities for each of the 7 categories
probabilities_df = pd.DataFrame(predictions, columns=[
    'nothing_prob', 
    'throw_wind_up_prob', 
    'throw_prob', 
    'throw_follow_through_prob', 
    'swing_wind_up_prob', 
    'swing_prob', 
    'swing_follow_through_prob'
])

# Round probabilities to 3 decimal places
probabilities_df = probabilities_df.round(3)
probabilities_df.to_csv('output_probabilities.csv', index=False)

# Initialize counters and tracking variables
num_swings = 0
num_throws = 0
ignore_count = 0  # To ignore the next two timestamps after detecting a swing or throw
increment_rows = []  # To store row numbers that triggered an increment

# Helper function to check neighboring rows for supporting probability
def has_supporting_probability(row_idx, category_indices, low_threshold):
    # Check if we have a previous row
    if row_idx > 0:
        prev_row = predictions[row_idx - 1]
        if any(prev_row[idx] > low_threshold for idx in category_indices):
            return True

    # Check if we have a next row
    if row_idx < len(predictions) - 1:
        next_row = predictions[row_idx + 1]
        if any(next_row[idx] > low_threshold for idx in category_indices):
            return True

    return False

# Process each row to apply the new logic
for i, row in enumerate(predictions):
    nothing_prob = row[0]
    throw_probs = row[1:4]  # throw_wind_up_prob, throw_prob, throw_follow_through_prob
    swing_probs = row[4:7]  # swing_wind_up_prob, swing_prob, swing_follow_through_prob

    # If in ignore mode, decrease the counter and continue to the next timestamp
    if ignore_count > 0:
        ignore_count -= 1
        continue

    # Check if the row should be labeled as "nothing"
    if nothing_prob > 0.98:
        continue

    # Check for swings with two conditions:
    # 1. If any swing probability > 0.90, check neighbors with threshold > 0.1
    # 2. If any swing probability is between 0.7 and 0.9, check neighbors with threshold > 0.25
    if any(prob > 0.90 for prob in swing_probs):
        if has_supporting_probability(i, range(4, 7), 0.1):
            num_swings += 1
            increment_rows.append(f"Swing detected at row {i}")
            ignore_count = 2
        continue
    elif any(0.7 < prob <= 0.9 for prob in swing_probs):
        if has_supporting_probability(i, range(4, 7), 0.25):
            num_swings += 1
            increment_rows.append(f"Swing detected at row {i} (low threshold)")
            ignore_count = 2
        continue

    # Check for throws with two conditions:
    # 1. If any throw probability > 0.90, check neighbors with threshold > 0.1
    # 2. If any throw probability is between 0.7 and 0.9, check neighbors with threshold > 0.25
    if any(prob > 0.90 for prob in throw_probs):
        if has_supporting_probability(i, range(1, 4), 0.1):
            num_throws += 1
            increment_rows.append(f"Throw detected at row {i}")
            ignore_count = 2
        continue
    elif any(0.7 < prob <= 0.9 for prob in throw_probs):
        if has_supporting_probability(i, range(1, 4), 0.25):
            num_throws += 1
            increment_rows.append(f"Throw detected at row {i} (low threshold)")
            ignore_count = 2
        continue

# Print the results
print(f"Number of swings: {num_swings}")
print(f"Number of throws: {num_throws}")

# Save the increment rows to a text file
with open("increment_rows.txt", "w") as f:
    for line in increment_rows:
        f.write(line + "\n")

