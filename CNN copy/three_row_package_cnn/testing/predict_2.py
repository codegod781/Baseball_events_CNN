import pandas as pd
import numpy as np
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # Import tqdm for progress bar



# Load the trained TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='model_fold_3.tflite')
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Pre-fit the scaler on the entire dataset using NumPy operations
def fit_scaler_on_test_data(data):
    # Prepare data sample for fitting scaler by creating combined rows with NumPy
    combined_samples = []
    for i in range(len(data) - 2):
        rows = data.iloc[i:i+3].values  # Convert to NumPy array for faster processing
        combined_row = []
        for z in range(0, 3):
            # Combine gyroscope data (each axis has 5 values)
            for j in range(1, 16):
                combined_row.append(rows[(j-1) // 5, 61+z + ((j-1)%5)*3])  # Gyro data
            # Combine accelerometer data (each axis has 20 values)
            for j in range(1, 61):
                combined_row.append(rows[(j-1) // 20, 1+z + ((j-1)%20)*3])  # Acc data
        combined_samples.append(combined_row)

    X_sample = np.array(combined_samples)
    scaler = StandardScaler()
    scaler.fit(X_sample)  # Fit on the entire sample
    return scaler

# Process and make predictions on the test data using NumPy and progress tracking
def predict_test_data(test_file, output_file, interpreter):
    # Load entire dataset into memory
    data = pd.read_csv(test_file)
    
    # Fit the scaler on the dataset
    scaler = fit_scaler_on_test_data(data)


    results = []
    
    # Use tqdm to show progress bar
    for i in tqdm(range(len(data) - 2), desc="Processing data", unit="sample"):
        # Extract data for 3 consecutive rows (1 second)
        rows = data.iloc[i:i+3].values  # Convert to NumPy array
        timestamp = rows[0, 0]  # Get timestamp from the first row

        combined_row = []
        for z in range(0, 3):
            # Combine gyroscope data (each axis has 5 values)
            for j in range(1, 16):
                combined_row.append(rows[(j-1) // 5, 61+z + ((j-1)%5)*3])  # Gyro data
            # Combine accelerometer data (each axis has 20 values)
            for j in range(1, 61):
                combined_row.append(rows[(j-1) // 20, 1+z + ((j-1)%20)*3])  # Acc data

        X_package = np.array([combined_row])
        X_package_scaled = scaler.transform(X_package)  # Transform with pre-fitted scaler


        # Reshape input to match TensorFlow Lite input format (usually NHWC for TFLite)
        X_package_scaled = X_package_scaled.reshape(1, X_package_scaled.shape[1], 1)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], X_package_scaled.astype(np.float32))

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Flatten and round the probabilities
        rounded_probabilities = [round(prob, 2) for prob in output_data.flatten()]

        # Store results
        results.append([timestamp] + rounded_probabilities)

    # Save results to a DataFrame and then to CSV
    results_df = pd.DataFrame(results, columns=['timestamp', 'prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4', 'prob_5', 'prob_6'])
    results_df.to_csv(output_file, index=False)


# Main function
def main():
    parser = argparse.ArgumentParser(description="Run TensorFlow Lite model on test data.")
    parser.add_argument("test_file", type=str, help="Path to the test data CSV file")
    parser.add_argument("--output_file", type=str, default="computed_probs.csv", help="Output file for probabilities (default: computed_probs.csv)")
    args = parser.parse_args()
    
    predict_test_data(args.test_file, args.output_file, interpreter)

if __name__ == "__main__":
    main()

