import subprocess
import sys
import os
import shutil

def run_main(original_file):
    # Ensure the original file exists
    if not os.path.isfile(original_file):
        print(f"Error: {original_file} does not exist.")
        sys.exit(1)

    # Get the base name of the file (without the extension)
    base_name = os.path.splitext(os.path.basename(original_file))[0]

    # Paths for the scripts
    format_script = os.path.join(os.getcwd(), "format_test_data.py")
    predict_script = os.path.join(os.getcwd(), "predict_2.py")
    count_script = os.path.join(os.getcwd(), "count.py")
    stats_script = os.path.join(os.getcwd(), "baseball_stats.py")

    # Step 1: Format the data
    subprocess.run(
        [sys.executable, format_script, original_file, "zeros"],
        check=True,
    )
    print("Generated: filtered_test_data.csv")
    print("Generated: zeroed_out_data.csv\n")

    # Step 2: Run predictions
    subprocess.run(
        [sys.executable, predict_script, "filtered_test_data.csv"],
        check=True,
    )
    print("Generated: computed_probs.csv\n")

    # Step 3: Count events
    subprocess.run(
        [sys.executable, count_script, "computed_probs.csv"],
        check=True,
    )
    print("Generated: event_log.txt")

    # Step 4: Calculate stats
    subprocess.run(
        [sys.executable, stats_script, "event_log.txt", "filtered_test_data.csv"],
        check=True,
    )
    print("\nGenerated: practice_stats.txt\n")

    # Create output directory
    output_dir = os.path.join(os.getcwd(), base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Move output files to the output directory
    output_files = [
        "filtered_test_data.csv",
        "zeroed_out_data.csv",
        "computed_probs.csv",
        "event_log.txt",
        "practice_stats.txt",
    ]
    for file in output_files:
        if os.path.exists(file):
            shutil.move(file, os.path.join(output_dir, file))

    print(f"Moved: output files to {output_dir}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <original_csv_file>")
        sys.exit(1)

    run_main(sys.argv[1])

