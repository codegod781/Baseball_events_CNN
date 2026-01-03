import pandas as pd
import numpy as np

# Constants
SAMPLE_INTERVAL = 0.05  # Gyroscope sampling interval (seconds)

def calculate_total_gyro_acc(data):
    """
    Calculate total gyroscope acceleration for each sample in the dataset.
    """
    total_gyro_acc = np.sqrt(
        data.filter(regex=r'gyro_x_\d').values**2 +
        data.filter(regex=r'gyro_y_\d').values**2 +
        data.filter(regex=r'gyro_z_\d').values**2
    )
    return total_gyro_acc

def process_events(event_log_file, imu_data_file, output_file):
    # Load event log and IMU data
    with open(event_log_file, 'r') as f:
        events = [line.strip().split(': ') for line in f.readlines()]

    imu_data = pd.read_csv(imu_data_file)
    total_gyro_acc = calculate_total_gyro_acc(imu_data)

    # Variables to track statistics
    throw_max_gyro_accs = []
    swing_max_gyro_accs = []
    high_intensity_throws = 0
    low_intensity_throws = 0
    high_intensity_swings = 0
    low_intensity_swings = 0
    event_details = []

    for timestamp_str, event_type in events:
        timestamp = float(timestamp_str)
        
        # Find the row index matching the timestamp
        matching_index = imu_data[imu_data['time_trame'] == timestamp].index
        if len(matching_index) == 0:
            print(f"Timestamp {timestamp} not found in IMU data.")
            continue
        
        matching_index = matching_index[0]
        
        # Determine the window of rows to examine
        start_index = max(0, matching_index - 3)
        end_index = min(len(imu_data), matching_index + 4)
        
        # Calculate the maximum total gyroscope acceleration in the window
        max_gyro_acc = total_gyro_acc[start_index:end_index].max()
        
        # Collect event details
        event_details.append((timestamp, event_type, max_gyro_acc))

        # Categorize events and track statistics
        if event_type == "Throw":
            throw_max_gyro_accs.append(max_gyro_acc)
            if max_gyro_acc > 1000:
                high_intensity_throws += 1
            else:
                low_intensity_throws += 1
        elif event_type == "Swing":
            swing_max_gyro_accs.append(max_gyro_acc)
            if max_gyro_acc > 900:
                high_intensity_swings += 1
            else:
                low_intensity_swings += 1

    # Calculate averages
    average_throw_gyro_acc = np.mean(throw_max_gyro_accs) if throw_max_gyro_accs else 0
    average_swing_gyro_acc = np.mean(swing_max_gyro_accs) if swing_max_gyro_accs else 0

    # Write summary statistics and event details to output file
    with open(output_file, 'w') as output:
        output.write("=== Event Statistics ===\n")
        output.write("Values are Total Gyroscopic Acceleration (m/s^2)\n\n")

        output.write("Summary Statistics:\n")
        output.write(f"  # Throws: {high_intensity_throws+low_intensity_throws}\n")
        output.write(f"  # Swings: {high_intensity_swings+low_intensity_swings}\n")
        output.write(f"  # High-Intensity Throws (>1000): {high_intensity_throws}\n")
        output.write(f"  # Low-Intensity Throws (<1000): {low_intensity_throws}\n")
        output.write(f"  # High-Intensity Swings (>900): {high_intensity_swings}\n")
        output.write(f"  # Low-Intensity Swings (<900): {low_intensity_swings}\n")
        output.write(f"    Average Throw Max Gyro Acc: {average_throw_gyro_acc:.2f}\n")
        output.write(f"    Average Swing Max Gyro Acc: {average_swing_gyro_acc:.2f}\n\n")

        output.write("Detailed Event List:\n")
        output.write(f"{'Timestamp |':<15} {'Event Type |':<10} {'Max Gyro Acc (m/s^2)':>20}\n")
        output.write("-" * 50 + "\n")
        for timestamp, event_type, max_gyro_acc in event_details:
            output.write(f"{timestamp:<15} {event_type:<10} {max_gyro_acc:>20.2f}\n")

    # Print summary statistics to the console
    print("\nSummary Statistics:")
    print(f"# Throws: {high_intensity_throws+low_intensity_throws}")
    print(f"# Swings: {high_intensity_swings+low_intensity_swings}")
    print(f"# High-Intensity Throws (>1000): {high_intensity_throws}")
    print(f"# Low-Intensity Throws (<1000): {low_intensity_throws}")
    print(f"# High-Intensity Swings (>900): {high_intensity_swings}")
    print(f"# Low-Intensity Swings (<900): {low_intensity_swings}")
    print(f"Average Throw Max Gyro Acc: {average_throw_gyro_acc:.2f}")
    print(f"Average Swing Max Gyro Acc: {average_swing_gyro_acc:.2f}")

# Main execution
if __name__ == "__main__":
    EVENT_LOG_FILE = "event_log.txt"
    IMU_DATA_FILE = "filtered_test_data.csv"
    OUTPUT_FILE = "practice_stats.txt"

    process_events(EVENT_LOG_FILE, IMU_DATA_FILE, OUTPUT_FILE)

