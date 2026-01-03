import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Load data
filename = sys.argv[1] if len(sys.argv) > 1 else 'grainular_training_data.csv'
data = pd.read_csv(filename)

# Time information
row_interval = 0.25  # seconds per row
sample_interval = row_interval / 5  # 0.05 seconds between samples

# Initialize time vectors
row_times = data['time_trame']
sample_times = np.arange(len(row_times) * 5) * sample_interval

# Calculate total acceleration for each gyroscope sample
total_gyro_acc = np.sqrt(
    data.filter(regex=r'gyro_x_\d').values**2 +
    data.filter(regex=r'gyro_y_\d').values**2 +
    data.filter(regex=r'gyro_z_\d').values**2
)

# Reshape the total gyroscope acceleration into a single 1D array
total_gyro_acc_flat = total_gyro_acc.flatten()

# Find points where acceleration exceeds threshold
threshold = 400
peak_indices = np.where(total_gyro_acc_flat > threshold)[0]
peak_times = sample_times[peak_indices]
peak_values = total_gyro_acc_flat[peak_indices]

# Create the plot
plt.figure(figsize=(12, 6))

# Plot the main acceleration data
plt.plot(sample_times, total_gyro_acc_flat, label='Total Gyroscope Acceleration')

# Plot threshold points
plt.scatter(peak_times, peak_values, color='red', marker='x', s=100, 
           label=f'Peaks > {threshold}')

# Add timestamp annotations for peaks
for time, value in zip(peak_times, peak_values):
    plt.annotate(f'{time:.2f}s',  # Format time to 2 decimal places
                xy=(time, value))

plt.xlabel('Time (s)')
plt.ylabel('Total Gyroscope Acceleration (m/s²)')
plt.title('Total Gyroscope Acceleration Over Time with Peak Detection')
plt.legend(loc="upper right")  # Set a fixed location for the legend
plt.grid()

# Add a horizontal line for the threshold
plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.3, 
           label=f'Threshold ({threshold})')

plt.show()

