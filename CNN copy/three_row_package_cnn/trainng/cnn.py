import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import os

# Load and preprocess the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    
    # Separate features and labels
    X = data.drop(columns=['label']).values
    y = data['label'].values
    
    # Normalize the feature values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape X for CNN input (samples, time steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Adding an extra dimension for CNN input
    y = to_categorical(y, num_classes=7)  # One-hot encode labels for 7 classes
    
    return X, y

# Define the CNN model
def create_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(256, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(7, activation='softmax')  # 7 output classes
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# K-fold cross-validation training
def train_model_kfold(X, y, k=1, epochs=10, batch_size=32, save_path="saved_models"):
    kfold = KFold(n_splits=k, shuffle=True, random_state=1)
    fold_no = 1
    all_scores = []

    # Ensure the directory exists for saving models
    os.makedirs(save_path, exist_ok=True)

    for train_index, val_index in kfold.split(X):
        print(f"Training on fold {fold_no}...")
        
        # Split data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Create a new instance of the model
        model = create_model(input_shape=(X.shape[1], 1))
        
        # Train the model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                            validation_data=(X_val, y_val), verbose=1)
        
        # Evaluate the model
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"Fold {fold_no} - Validation Accuracy: {val_acc}")
        all_scores.append(val_acc)
        
        # Print classification report for each fold
        y_pred = np.argmax(model.predict(X_val), axis=1)
        y_true = np.argmax(y_val, axis=1)
        print(classification_report(y_true, y_pred, target_names=['Nothing', 'Throw Wind Up', 'Throw', 
                                                                  'Throw Follow Through', 'Swing Load Up', 
                                                                  'Swing', 'Swing Follow Through']))

        # Save the model for each fold
        model_save_path = os.path.join(save_path, f'model_fold_{fold_no}.h5')
        model.save(model_save_path)
        print(f"Model for fold {fold_no} saved at {model_save_path}")

        fold_no += 1

    # Print overall performance
    print(f"Average Accuracy across {k} folds: {np.mean(all_scores)}")
    print(f"Standard Deviation of Accuracy across folds: {np.std(all_scores)}")

# Main execution
input_file = 'three_row_packages.csv'  # Update this path to your combined data file
X, y = load_data(input_file)
train_model_kfold(X, y, k=5, epochs=20, batch_size=32)

