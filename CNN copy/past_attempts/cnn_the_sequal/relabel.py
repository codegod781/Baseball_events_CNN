import pandas as pd

# Load your labeled data
label_data = pd.read_csv('total_training_data.csv')

# Print column names to check if 'label' is correct
print("Columns in the dataset:", label_data.columns)

# Assuming the correct column name is 'label' instead of 'event'
new_labels = label_data['label'].copy()  # Start with the existing labels

# Loop through the rows and relabel throws and swings as sequences
split_label = False
for idx in range(len(label_data) - 2):
    current_label = label_data.iloc[idx]['label']
    if split_label == True :
        split_label = False
        continue
    # Handle throw (label 1)
    if current_label == 1:  # Assuming 1 is a throw
        if label_data.iloc[idx + 1]['label'] == 1:  # Check if throw spans two timestamps
            new_labels.iloc[idx - 1] = 1  # Wind-up for throw
            new_labels.iloc[idx] = 2  # Throw
            new_labels.iloc[idx + 1] = 2  # Throw
            new_labels.iloc[idx + 2] = 3  # Follow-through
            split_label = True
        else:
            new_labels.iloc[idx-1] = 1  # Wind-up for throw
            new_labels.iloc[idx] = 2  # Throw
            new_labels.iloc[idx + 1] = 3  # Follow-through for throw

    # Handle swing (label 2)
    elif current_label == 2:  # Assuming 2 is a swing
        if label_data.iloc[idx + 1]['label'] == 2:  # Check if swing spans two timestamps
            new_labels.iloc[idx - 1] = 4  # Wind-up for swing
            new_labels.iloc[idx] = 5  # Swing
            new_labels.iloc[idx + 1] = 5  # Swing
            new_labels.iloc[idx + 2] = 6  # Follow-through
            split_label = True
        else:
            new_labels.iloc[idx-1] = 4  # Wind-up for swing
            new_labels.iloc[idx] = 5  # Swing
            new_labels.iloc[idx + 1] = 6  # Follow-through for swing

# Replace the old labels with the new labels
label_data['label'] = new_labels

# Save the updated labels
label_data.to_csv('updated_label_data.csv', index=False)

