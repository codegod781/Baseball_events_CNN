import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Load the CSV data
file_path = 'total_filtered_grainular_training_data.csv'  # Update with the path to your file if necessary
data = pd.read_csv(file_path)

# Prepare the dataset
data = data.dropna(subset=['label'])
X = data.drop(columns=['time_trame', 'label'])
y = data['label'].astype(int)



# Normalize the sensor data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the data to fit into a CNN (samples, time steps, features)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Convert labels to categorical (one-hot encoding)
y_categorical = tf.keras.utils.to_categorical(y, num_classes=7)

# Set up k-fold cross-validation
k = 5  # You can adjust the number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Variables to store accuracy for each fold
fold_accuracies = []

for train_index, test_index in kf.split(X_reshaped):
    # Split the data into training and testing sets for this fold
    X_train, X_test = X_reshaped[train_index], X_reshaped[test_index]
    y_train, y_test = y_categorical[train_index], y_categorical[test_index]
    
    # Define the CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=300, kernel_size=1, activation='relu', input_shape=(1, X_reshaped.shape[2])),
        tf.keras.layers.MaxPooling1D(pool_size=1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')  # 7 classes: nothing, throw wind up, throw, throw follow through, swing wind up, swing, swing follow through
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    
    # Evaluate the model on the test set for this fold
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Fold Test Accuracy: {test_acc * 100:.2f}%")
    fold_accuracies.append(test_acc)

# Calculate and print the average accuracy across all folds
average_accuracy = np.mean(fold_accuracies)
print(f"Average Cross-Validation Accuracy: {average_accuracy * 100:.2f}%")

# Save the model if you want to reuse it later
model.save("sports_tracker_model.h5")

