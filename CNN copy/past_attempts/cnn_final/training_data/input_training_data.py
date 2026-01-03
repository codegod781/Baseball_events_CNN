import pandas as pd
import os
import sys



def append_to_total_training_data(file_to_append):
    # Define the name of the total training data file
    total_training_file = 'total_training_data.csv'
    
    # Read and filter the file to append
    df_to_append = pd.read_csv(file_to_append, delimiter=';', quotechar='"', header=0)
    print("Columns in the loaded DataFrame:", df_to_append.columns.tolist())  # Check columns here
    df_to_append = filter_columns(df_to_append)
    #print("Columns after filtering:", df_to_append.columns)

    # Apply the transformation
    subrows_df = split_and_reorder_subrows(df_to_append)
    subrows_df.to_csv("grainular_training_data.csv", mode='a', index=False, header = 1)  # Save the result


    # Determine starting line for the new data
    starting_line = get_starting_line(total_training_file)

    # Append data to the total training data file
    df_to_append.to_csv(total_training_file, mode='a', index=False, header=0)
    print(f"{file_to_append} appended to total_training_data.csv.")

    # Calculate ending line and update ledger
    ending_line = starting_line + len(df_to_append) - 1
    update_ledger(file_to_append, starting_line, ending_line)



def filter_columns(df):
    cols_to_drop = ['num_sv', 'hdop', 'error_gps', 'bpm', 'error_cardio', 
                    'is_in_charge', 'battery', 
                    'latitude_1', 'latitude_2', 'latitude_3', 'latitude_4', 
                    'latitude_5', 'latitude_6', 'latitude_7', 'latitude_8', 
                    'latitude_9', 'latitude_10', 
                    'longitude_1', 'longitude_2', 'longitude_3', 'longitude_4', 
                    'longitude_5', 'longitude_6', 'longitude_7', 'longitude_8', 
                    'longitude_9', 'longitude_10', 
                    'speed_1', 'speed_2', 'speed_3', 'speed_4', 
                    'speed_5', 'speed_6', 'speed_7', 'speed_8', 
                    'speed_9', 'speed_10']
    cols_to_drop += [f'mag_x_{i}' for i in range(1, 21)]  # Assuming there are 20 magnetometer columns
    cols_to_drop += [f'mag_y_{i}' for i in range(1, 21)]
    cols_to_drop += [f'mag_z_{i}' for i in range(1, 21)]
    df = df.drop(columns=cols_to_drop, errors='ignore')  # 'errors=ignore' skips missing columns
    return df

import pandas as pd




import pandas as pd

def split_and_reorder_subrows(df):
    # Prepare a list to hold the new rows
    subrows = []
    
    for _, row in df.iterrows():
        # Create each subrow with reordered columns
        subrow_data = [
            # Subrow 1: First 20 accelerometer and first 5 gyroscope values, reordered
            {f"{axis}_{i}": row[f"{axis}_{i}"] for i in range(1, 21) for axis in ["acc_x", "acc_y", "acc_z"]} |
            {f"gyro_{axis}_{i}": row[f"gyro_{axis}_{i}"] for i in range(1, 6) for axis in ["x", "y", "z"]},
            
            # Subrow 2: Next 20 accelerometer and 5 gyroscope values, reordered
            {f"{axis}_{i}": row[f"{axis}_{i+20}"] for i in range(1, 21) for axis in ["acc_x", "acc_y", "acc_z"]} |
            {f"gyro_{axis}_{i}": row[f"gyro_{axis}_{i+5}"] for i in range(1, 6) for axis in ["x", "y", "z"]},
            
            # Subrow 3: Following 20 accelerometer and 5 gyroscope values, reordered
            {f"{axis}_{i}": row[f"{axis}_{i+40}"] for i in range(1, 21) for axis in ["acc_x", "acc_y", "acc_z"]} |
            {f"gyro_{axis}_{i}": row[f"gyro_{axis}_{i+10}"] for i in range(1, 6) for axis in ["x", "y", "z"]},
            
            # Subrow 4: Final 20 accelerometer and 5 gyroscope values, reordered
            {f"{axis}_{i}": row[f"{axis}_{i+60}"] for i in range(1, 21) for axis in ["acc_x", "acc_y", "acc_z"]} |
            {f"gyro_{axis}_{i}": row[f"gyro_{axis}_{i+15}"] for i in range(1, 6) for axis in ["x", "y", "z"]}
        ]
        
        # Append each of these subrows to the list
        subrows.extend(subrow_data)

    # Create a new DataFrame from these subrows with the reordered structure
    subrows_df = pd.DataFrame(subrows)
    
    # Reorder columns to have each index grouped together (e.g., acc_x_1, acc_y_1, acc_z_1, etc.)
    reordered_columns = [f"{axis}_{i}" for i in range(1, 21) for axis in ["acc_x", "acc_y", "acc_z"]] + \
                        [f"gyro_{axis}_{i}" for i in range(1, 6) for axis in ["x", "y", "z"]]
    subrows_df = subrows_df[reordered_columns]
    
    # Add 'time_frame' and 'label' columns
    subrows_df.insert(0, 'time_trame', [i * 0.25 for i in range(len(subrows_df))])
    subrows_df.insert(1, 'label', 0)
    
    return subrows_df








def get_starting_line(total_training_file):
    """Calculate the starting line for new data in the total training file."""
    if os.path.exists(total_training_file):
        with open(total_training_file, 'r') as f:
            return sum(1 for _ in f) + 1  # Existing lines + 1 for 0-index
    else:
        return 1  # Start from line 1 if the file does not exist


def update_ledger(file_to_append, starting_line, ending_line):
    """Update the ledger with the filename and line range of appended data."""
    ledger_file = 'order_training_data.txt'
    with open(ledger_file, 'a') as ledger:
        ledger.write(f"{file_to_append} {starting_line}-{ending_line}\n")
    print(f"Ledger updated with range {starting_line}-{ending_line} for {file_to_append}.")

if __name__ == "__main__":
    # Check if a filename is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python input_training_data.py <filename>")
        sys.exit(1)

    # Get the filename from the command line arguments
    file_to_append = sys.argv[1]

    # Append the file to total_training_data.csv
    append_to_total_training_data(file_to_append)
    print("remember to adjust times in grainular")

