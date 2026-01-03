import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load reshaped gyroscope data
df_gyro = pd.read_csv('reshaped_gyro_data.csv')

# Assuming the data is sampled at 20Hz (i.e., every 0.05 seconds)
sampling_rate = 20  # 20 Hz = 0.05 seconds per sample
df_gyro['time'] = np.arange(len(df_gyro)) / sampling_rate

# Calculate total acceleration as the magnitude of the three components
df_gyro['total_acceleration'] = np.sqrt(df_gyro['gyro_x']**2 + df_gyro['gyro_y']**2 + df_gyro['gyro_z']**2)

# Identify timestamps where total acceleration exceeds 500 deg/s
high_accel_threshold = 300
high_accel_points = df_gyro[df_gyro['total_acceleration'] > high_accel_threshold]

# Plot total gyroscope acceleration over time
plt.figure(figsize=(10, 6))
plt.plot(df_gyro['time'], df_gyro['total_acceleration'], color='purple', label='Total Acceleration')

# Mark points where acceleration exceeds the threshold
plt.scatter(high_accel_points['time'], high_accel_points['total_acceleration'], color='red', label='> 500 deg/s', zorder=5)

# Annotate times where acceleration exceeds the threshold
for idx, row in high_accel_points.iterrows():
    plt.annotate(f"{row['time']:.2f}s", (row['time'], row['total_acceleration']),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='black')

# Customize plot labels and title
plt.title('Total Gyroscope Acceleration Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Total Acceleration (deg/s)')

# Add a legend
plt.legend()

# Save the plot as an image file (e.g., PNG)
plt.savefig('gyroscope_total_acceleration_plot_with_times.png')

# Show the plot
plt.show()

