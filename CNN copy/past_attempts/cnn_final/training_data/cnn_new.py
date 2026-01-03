import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def extract_packages(input_file):
    # Load the data
    data = pd.read_csv(input_file)
    
    # Prepare lists to collect packages and labels
    packaged_data = []
    labels = []

    # Iterate through the rows to find throws and swings
    for i in range(len(data)):
        # Check for a throw (label = 1)
        if data.at[i, 'label'] == 1:
            # Define the indices for the throw package (4 rows before, 7 after)
            start_idx = max(0, i - 4)
            end_idx = i + 8
            
            if end_idx <= len(data):  # Ensure we don't exceed the data length
                package = data.iloc[start_idx:end_idx]
                
                # Verify package length to be exactly 12 rows
                if len(package) == 12:
                    # Extract gyro and accel data, preserving time order
                    gyro_data = package.filter(regex='gyro').to_numpy()
                    accel_data = package.filter(regex='acc').to_numpy()
                    
                    # Combine gyro and accel data into a single 2D array
                    package_2d = np.hstack([gyro_data, accel_data])
                    packaged_data.append(package_2d)
                    
                    # Label the package as a throw
                    labels.append('throw')

        # Check for a swing (label = 4)
        elif data.at[i, 'label'] == 4:
            # Define the indices for the swing package (2 rows before, 9 after)
            start_idx = max(0, i - 2)
            end_idx = i + 10
            
            if end_idx <= len(data):  # Ensure we don't exceed the data length
                package = data.iloc[start_idx:end_idx]
                
                # Verify package length to be exactly 12 rows
                if len(package) == 12:
                    # Extract gyro and accel data, preserving time order
                    gyro_data = package.filter(regex='gyro').to_numpy()
                    accel_data = package.filter(regex='acc').to_numpy()
                    
                    # Combine gyro and accel data into a single 2D array
                    package_2d = np.hstack([gyro_data, accel_data])
                    packaged_data.append(package_2d)
                    
                    # Label the package as a swing
                    labels.append('swing')

    # Convert labels to numeric values (e.g., 0 for 'throw', 1 for 'swing')
    label_encoder = LabelEncoder()
    labels_numeric = label_encoder.fit_transform(labels)

    # Convert packaged data to a NumPy array
    X = np.array(packaged_data)
    
    print("Shape of X:", X.shape)  # Should be (499, 12, 75)
    
    # Add channel dimension for CNN
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    print("Final shape of X:", X.shape)  # Should be (499, 12, 75, 1)

    # Convert labels to a NumPy array
    y = np.array(labels_numeric)

    return X, y

# Example usage: Extract data and prepare it for the CNN model
X, y = extract_packages('labeled_grainular_training_data.csv')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = tf.keras.Sequential([
    # First convolutional block
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(12, 75, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    
    # Second convolutional block
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
    
    # Flatten and dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Add model summary to see the layer dimensions
model.summary()

# Train the model
history = model.fit(
    X_train, 
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

#Save the model
model.save('imu_event_classifier.h5')  # Saves in HDF5 format


