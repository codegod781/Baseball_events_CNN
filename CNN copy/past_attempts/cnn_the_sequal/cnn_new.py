import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the CSV data
file_path = 'updated_label_data.csv'  # Update with the path to your file if necessary
data = pd.read_csv(file_path)

# Prepare the dataset
X = data.drop(columns=['time_trame', 'label'])
y = data['label']

# Normalize the sensor data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the data to fit into a CNN (samples, time steps, features)
# Here, we will ensure that the number of features aligns with the model's expectations
num_features = X_scaled.shape[1]  # Number of features in the input
X_reshaped = X_scaled.reshape((-1, 2, num_features // 2))  # Ensure we get the correct shape

# Convert labels to categorical (one-hot encoding)
y_categorical = tf.keras.utils.to_categorical(y, num_classes=7)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

# Define the CNN model using Input layer
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2, num_features // 2)),  # Adjust based on how you reshape
    tf.keras.layers.Conv1D(filters=300, kernel_size=2, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')  # 7 classes: nothing, throw wind up, throw, throw follow through, swing wind up, swing, swing follow through
])

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model if you want to reuse it later
model.save("sports_tracker_model.h5")

