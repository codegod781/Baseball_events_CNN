import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the CSV data
file_path = 'total_training_data.csv'  # Update with the path to your file if necessary
data = pd.read_csv(file_path)

# Prepare the dataset
# Remove the 'time_trame' column and separate features and labels
X = data.drop(columns=['time_trame', 'label'])
y = data['label']

# Normalize the sensor data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the data to fit into a CNN (samples, time steps, features)
# Each row represents 1 second, so we'll reshape to (samples, 1, features)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Convert labels to categorical (one-hot encoding)
y_categorical = tf.keras.utils.to_categorical(y, num_classes=3)
print(y_categorical)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

# Define the CNN model with kernel size of 1
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=340, kernel_size=1, activation='relu', input_shape=(1, X_reshaped.shape[2])),
    tf.keras.layers.MaxPooling1D(pool_size=1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: nothing (0), throw (1), swing (2)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open("sports_tracker_model.tflite", "wb") as f:
    f.write(tflite_model)

