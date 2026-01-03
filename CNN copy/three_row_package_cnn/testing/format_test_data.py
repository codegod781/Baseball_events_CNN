import pandas as pd
import os
import sys
import numpy as np

def append_to_total_training_data(file_to_append, save_zeros=False):
    df_to_append = pd.read_csv(file_to_append, delimiter=';', quotechar='"', header=0)
#    print(df_to_append.columns)

    df_to_append = filter_columns(df_to_append)
    # Turn time frames from 1s to 0.25s
    subrows_df = split_and_reorder_subrows(df_to_append)
    
    zeroed_df = zero_out_low_activity_intervals(subrows_df.copy())
    if save_zeros:
        zeroed_df.to_csv("zeroed_out_data.csv", index=False)
    filtered_df = delete_low_activity_intervals(zeroed_df)

    filtered_df.to_csv("filtered_test_data.csv", mode='w', index=False, header=True)


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
    return subrows_df


def zero_out_low_activity_intervals(df, threshold=350, interval_length=20):
    total_gyro_acc = np.sqrt(
        df.filter(regex=r'gyro_x_\d').values**2 +
        df.filter(regex=r'gyro_y_\d').values**2 +
        df.filter(regex=r'gyro_z_\d').values**2
    )

    rows_per_interval = int(interval_length / 0.25)
    for i in range(0, len(df), rows_per_interval):
        interval_slice = slice(i, i + rows_per_interval)
        
        # Check if any value in the interval exceeds the threshold
        if not np.any(total_gyro_acc[interval_slice] > threshold):
            for axis in ["x", "y", "z"]:
                for j in range(1, 6):
                    df.loc[interval_slice, f"gyro_{axis}_{j}"] = 0

    return df


def delete_low_activity_intervals(df, threshold=350, interval_length=20):
    total_gyro_acc = np.sqrt(
        df.filter(regex=r'gyro_x_\d').values**2 +
        df.filter(regex=r'gyro_y_\d').values**2 +
        df.filter(regex=r'gyro_z_\d').values**2
    )

    rows_per_interval = int(interval_length / 0.25)
    rows_to_keep = []

    for i in range(0, len(df), rows_per_interval):
        interval_slice = slice(i, i + rows_per_interval)
        if np.any(total_gyro_acc[interval_slice] > threshold):
            rows_to_keep.extend(range(i, i + rows_per_interval))
    
    df = df.iloc[rows_to_keep].reset_index(drop=True)
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python format_test_data.py <filename> [zeros]")
        sys.exit(1)

    file_to_append = sys.argv[1]
    save_zeros = len(sys.argv) > 2 and sys.argv[2] == "zeros"

    append_to_total_training_data(file_to_append, save_zeros)

