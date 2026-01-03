import pandas as pd
from collections import Counter

def combine_rows(input_file, output_file):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(input_file)
    
    # Initialize a list to store combined rows
    combined_data = []
    
    # Loop through the DataFrame in a sliding window of three rows
    for i in range(len(data) - 2):
        # Extract the three rows
        rows = data.iloc[i:i+3]
        
        # Combine gyroscope and accelerometer data for these rows
        combined_row = {}
        for axis in ['x', 'y', 'z']:
            # Combine gyroscope data
            for j in range(1, 16):  # 15 gyroscope columns
                combined_row[f'gyro_{axis}_{j}'] = rows[f'gyro_{axis}_{((j - 1) % 5) + 1}'].values[(j - 1) // 5]
            
            # Combine accelerometer data
            for j in range(1, 61):  # 60 accelerometer columns
                combined_row[f'acc_{axis}_{j}'] = rows[f'acc_{axis}_{((j - 1) % 20) + 1}'].values[(j - 1) // 20]
        
        # Determine the label by majority rule
        labels = rows['label'].tolist()
        majority_label = Counter(labels).most_common(1)[0][0]
        combined_row['label'] = majority_label

        # Add the combined row to the list
        combined_data.append(combined_row)

    # Convert the list of combined rows to a DataFrame
    combined_df = pd.DataFrame(combined_data)
    
    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")

# Usage
input_file = 'updated_training_data.csv'   # Replace with your input file path
output_file = 'three_row_packages.csv'   # Replace with your desired output file path
combine_rows(input_file, output_file)

