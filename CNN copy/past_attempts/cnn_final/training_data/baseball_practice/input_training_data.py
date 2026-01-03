import pandas as pd
import os
import sys
import numpy as np

def append_to_total_training_data(file_to_append):
    total_training_file = 'total_training_data.csv'
    
    df_to_append = pd.read_csv(file_to_append, delimiter=';', quotechar='"', header=0)
    df_to_append = filter_columns(df_to_append)

    subrows_df = split_and_reorder_subrows(df_to_append)
    subrows_df.to_csv("grainular_training_data.csv", mode='a', index=False, header=1)
    # Apply the two filters
    subrows_df = zero_out_low_activity_intervals(subrows_df)
    subrows_df = zero_out_high_activity_intervals(subrows_df)  # New function
    subrows_df.to_csv("filtered_grainular_training_data.csv", mode='a', index=False, header=0)



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
    cols_to_drop += [f'mag_x_{i}' for i in range(1, 21)]
    cols_to_drop += [f'mag_y_{i}' for i in range(1, 21)]
    cols_to_drop += [f'mag_z_{i}' for i in range(1, 21)]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df


def split_and_reorder_subrows(df):
    subrows = []
    
    for _, row in df.iterrows():
        subrow_data = [
            {f"{axis}_{i}": row[f"{axis}_{i}"] for i in range(1, 21) for axis in ["acc_x", "acc_y", "acc_z"]} |
            {f"gyro_{axis}_{i}": row[f"gyro_{axis}_{i}"] for i in range(1, 6) for axis in ["x", "y", "z"]},
            
            {f"{axis}_{i}": row[f"{axis}_{i+20}"] for i in range(1, 21) for axis in ["acc_x", "acc_y", "acc_z"]} |
            {f"gyro_{axis}_{i}": row[f"gyro_{axis}_{i+5}"] for i in range(1, 6) for axis in ["x", "y", "z"]},
            
            {f"{axis}_{i}": row[f"{axis}_{i+40}"] for i in range(1, 21) for axis in ["acc_x", "acc_y", "acc_z"]} |
            {f"gyro_{axis}_{i}": row[f"gyro_{axis}_{i+10}"] for i in range(1, 6) for axis in ["x", "y", "z"]},
            
            {f"{axis}_{i}": row[f"{axis}_{i+60}"] for i in range(1, 21) for axis in ["acc_x", "acc_y", "acc_z"]} |
            {f"gyro_{axis}_{i}": row[f"gyro_{axis}_{i+15}"] for i in range(1, 6) for axis in ["x", "y", "z"]}
        ]
        
        subrows.extend(subrow_data)

    subrows_df = pd.DataFrame(subrows)
    
    reordered_columns = [f"{axis}_{i}" for i in range(1, 21) for axis in ["acc_x", "acc_y", "acc_z"]] + \
                        [f"gyro_{axis}_{i}" for i in range(1, 6) for axis in ["x", "y", "z"]]
    subrows_df = subrows_df[reordered_columns]
    
    subrows_df.insert(0, 'time_trame', [i * 0.25 for i in range(len(subrows_df))])
    subrows_df.insert(1, 'label', 0)
    
    return subrows_df


def zero_out_low_activity_intervals(df, threshold=350, interval_length=5):
    total_gyro_acc = np.sqrt(
        df.filter(regex=r'gyro_x_\d').values**2 +
        df.filter(regex=r'gyro_y_\d').values**2 +
        df.filter(regex=r'gyro_z_\d').values**2
    )

    rows_per_interval = int(interval_length / 0.25)
    
    for i in range(0, len(df), rows_per_interval):
        interval_slice = slice(i, i + rows_per_interval)
        
        if not np.any(total_gyro_acc[interval_slice] > threshold):
            for axis in ["x", "y", "z"]:
                for j in range(1, 6):
                    df.loc[interval_slice, f"gyro_{axis}_{j}"] = 0

    return df


def zero_out_high_activity_intervals(df, threshold=400, max_points=10, interval_length=5):
    total_gyro_acc = np.sqrt(
        df.filter(regex=r'gyro_x_\d').values**2 +
        df.filter(regex=r'gyro_y_\d').values**2 +
        df.filter(regex=r'gyro_z_\d').values**2
    )

    rows_per_interval = int(interval_length / 0.25)
    
    for i in range(0, len(df), rows_per_interval):
        interval_slice = slice(i, i + rows_per_interval)
        
        # Count samples above threshold in this interval
        if np.sum(total_gyro_acc[interval_slice] > threshold) > max_points:
            for axis in ["x", "y", "z"]:
                for j in range(1, 6):
                    df.loc[interval_slice, f"gyro_{axis}_{j}"] = 0

    return df



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python input_training_data.py <filename>")
        sys.exit(1)

    file_to_append = sys.argv[1]
    append_to_total_training_data(file_to_append)

