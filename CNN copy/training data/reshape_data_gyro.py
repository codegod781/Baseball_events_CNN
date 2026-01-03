import pandas as pd

# Load the CSV file
df_gyro = pd.read_csv('40_iso_gyro.csv')

# Reshape the data: convert each set of gyro_x, gyro_y, gyro_z columns to individual rows
reshaped_data = []

# Iterate through each row of the original dataframe
for idx, row in df_gyro.iterrows():
    # For each timestamp, create 20 new rows, one for each set of x, y, z values
    for i in range(1, 21):
        reshaped_data.append({
            'Timestamp': row['time_trame'] + (i * 50),  # Add microsecond offset (1/20 sec for 20Hz)
            'gyro_x': row[f'gyro_x_{i}'],
            'gyro_y': row[f'gyro_y_{i}'],
            'gyro_z': row[f'gyro_z_{i}']
        })

# Convert the reshaped list into a DataFrame
df_gyro_reshaped = pd.DataFrame(reshaped_data)

# Display the reshaped data
print(df_gyro_reshaped.head())

# Save the reshaped data to a new CSV file if needed
df_gyro_reshaped.to_csv('reshaped_gyro_data.csv', index=False)

