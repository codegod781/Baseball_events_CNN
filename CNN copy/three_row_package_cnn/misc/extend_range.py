import pandas as pd

def update_labels(input_file, output_file):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(input_file)
    
    # Ensure the label column exists
    if 'label' not in data.columns:
        raise ValueError("The CSV file must contain a 'label' column.")
    
    # Convert labels to a list for easy manipulation
    labels = data['label'].tolist()
    
    # Loop through the labels list and apply both transformations
    i = 0
    while i < len(labels):
        if labels[i] == 4:
            # Update for swing load up and follow through sequence
            for j in range(max(0, i - 4), i):
                labels[j] = 4
            for j in range(i, min(i + 4, len(labels))):
                labels[j] = 5
            for j in range(i + 4, min(i + 10, len(labels))):
                labels[j] = 6
            i += 9  # Skip over the modified sequence

        elif labels[i] == 1:
            # Update for throw windup and follow through sequence
            for j in range(max(0, i - 6), i):
                labels[j] = 1
            for j in range(i, min(i + 4, len(labels))):
                labels[j] = 2
            for j in range(i + 4, min(i + 10, len(labels))):
                labels[j] = 3
            i += 9  # Skip over the modified sequence
            
        else:
            i += 1

    # Update the label column in the DataFrame
    data['label'] = labels
    
    # Write the modified DataFrame to a new CSV file
    data.to_csv(output_file, index=False)
    print(f"Updated labels saved to {output_file}")

# Usage
input_file = 'labeled_grainular_training_data.csv'   # Replace with your input file path
output_file = 'updated_training_data.csv'   # Replace with your desired output file path
update_labels(input_file, output_file)

