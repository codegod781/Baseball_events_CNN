import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the CNN model
model = load_model('imu_event_classifier.h5')

# Load the test data
test_data = pd.read_csv('total_test_data.csv')

# Initialize a list to store the package probabilities
package_probabilities = []

# Iterate through the test data in steps of 1 row
for i in range(len(test_data) - 11):  # Subtract 11 to ensure we have 12 rows for each package
    # Extract the current package (12 rows)
    package = test_data.iloc[i:i+12]
    
    # Extract gyro and accel data, preserving time order
    gyro_data = package.filter(regex='gyro').to_numpy()
    accel_data = package.filter(regex='acc').to_numpy()
    
    # Combine gyro and accel data into a single 2D array
    package_2d = np.hstack([gyro_data, accel_data])
    
    # Add channel dimension for CNN
    package_2d = package_2d.reshape(1, package_2d.shape[0], package_2d.shape[1], 1)
    
    # Make predictions on the package
    predictions = model.predict(package_2d)
    
    # Extract the probabilities for each class (swing and throw)
    swing_prob = predictions[0][0]
    throw_prob = predictions[0][1]
    
    # Append the probabilities to the list
    package_probabilities.append([swing_prob, throw_prob])
    print("i : ", i)

# Convert the list to a NumPy array
package_probabilities = np.array(package_probabilities)

# Save the probabilities to a CSV file
np.savetxt('package_probability.csv', package_probabilities, delimiter=',')

print("Probabilities saved to package_probability.csv")
