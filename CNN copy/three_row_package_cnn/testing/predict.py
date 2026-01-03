import pandas as pd
import numpy as np
import argparse
import os
import tensorflow as tf  # Import TensorFlow Lite interpreter
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs except errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load the trained TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='model_fold_3.tflite')
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Pre-fit the scaler on the entire dataset
def fit_scaler_on_test_data(data):
    combined_samples = []
    for i in range(len(data) - 2):
        rows = data.iloc[i:i+3]
        combined_row = {}
        for axis in ['x', 'y', 'z']:
            # Combine gyroscope data
            for j in range(1, 16):
                combined_row[f'gyro_{axis}_{j}'] = rows[f'gyro_{axis}_{((j - 1) % 5) + 1}'].values[(j - 1) // 5]
            # Combine accelerometer data
            for j in range(1, 61):
                combined_row[f'acc_{axis}_{j}'] = rows[f'acc_{axis}_{((j - 1) % 20) + 1}'].values[(j - 1) // 20]
        combined_samples.append(combined_row)

    X_sample = pd.DataFrame(combined_samples).values
    scaler = StandardScaler()
    scaler.fit(X_sample)  # Fit on the entire sample
    return scaler

# Process and make predictions on the test data
def predict_test_data(test_file, output_file, interpreter, batch_size=64):
    # Load entire dataset into memory
    data = pd.read_csv(test_file)
    
    # Fit the scaler on the dataset
    scaler = fit_scaler_on_test_data(data)
    results = []

    for i in range(len(data) - 2):
        rows = data.iloc[i:i+3]
        timestamp = rows.iloc[0]['time_trame']

        combined_row = {}
        for axis in ['x', 'y', 'z']:
            for j in range(1, 16):
                combined_row[f'gyro_{axis}_{j}'] = rows[f'gyro_{axis}_{((j - 1) % 5) + 1}'].values[(j - 1) // 5]
            for j in range(1, 61):
                combined_row[f'acc_{axis}_{j}'] = rows[f'acc_{axis}_{((j - 1) % 20) + 1}'].values[(j - 1) // 20]

        X_package = pd.DataFrame([combined_row]).values
        X_package = scaler.transform(X_package)  # Transform with pre-fitted scaler
        
        # Reshape input to match TensorFlow Lite input format (usually NHWC for TFLite)
        X_package = X_package.reshape(1, X_package.shape[1], 1)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], X_package.astype(np.float32))

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Flatten and round the probabilities
        rounded_probabilities = [round(prob, 2) for prob in output_data.flatten()]

        results.append([timestamp] + rounded_probabilities)

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

