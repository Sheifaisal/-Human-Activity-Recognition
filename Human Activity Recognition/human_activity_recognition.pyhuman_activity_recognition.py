# human_activity_recognition.py
# A simple Human Activity Recognition (HAR) project using a Random Forest Classifier.
# This script simulates a dataset and trains a model to classify activities.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random


def generate_mock_data(num_samples=1000):
    """
    Generates a mock dataset for Human Activity Recognition.
    Features are simulated sensor readings (accelerometer, gyroscope).
    Labels are common activities.
    """
    activities = {
        'Walking': {'accel_x': (0.1, 0.5), 'accel_y': (0.5, 0.9), 'accel_z': (0.1, 0.4), 'gyro_x': (0.0, 0.2)},
        'Jogging': {'accel_x': (0.3, 0.8), 'accel_y': (0.7, 1.2), 'accel_z': (0.2, 0.6), 'gyro_x': (0.1, 0.4)},
        'Sitting': {'accel_x': (-0.1, 0.1), 'accel_y': (-0.1, 0.1), 'accel_z': (0.8, 1.0), 'gyro_x': (-0.1, 0.1)},
        'Standing': {'accel_x': (-0.1, 0.1), 'accel_y': (0.0, 0.2), 'accel_z': (0.9, 1.1), 'gyro_x': (-0.1, 0.1)},
        'Lying': {'accel_x': (-0.2, 0.2), 'accel_y': (-0.2, 0.2), 'accel_z': (-0.1, 0.1), 'gyro_x': (-0.1, 0.1)}
    }

    data = []
    labels = []

    # Generate synthetic sensor data
    for activity, ranges in activities.items():
        for _ in range(num_samples // len(activities)):
            sample = {
                'accel_x': random.uniform(*ranges['accel_x']),
                'accel_y': random.uniform(*ranges['accel_y']),
                'accel_z': random.uniform(*ranges['accel_z']),
                'gyro_x': random.uniform(*ranges['gyro_x']),
                'gyro_y': random.uniform(-0.5, 0.5),  # noise
                'gyro_z': random.uniform(-0.5, 0.5)   # noise
            }
            data.append(sample)
            labels.append(activity)

    df = pd.DataFrame(data)
    df['activity'] = labels

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def main():
    print("🚀 Starting Human Activity Recognition project...")

    # Step 1: Data
    print("Generating mock HAR data...")
    df = generate_mock_data()
    print("✅ Data generated successfully.\n")
    print(df.head())
    print("\nDataset shape:", df.shape)

    # Step 2: Features & Target
    X = df.drop('activity', axis=1)
    y = df['activity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # Step 3: Train model
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    # Step 4: Evaluation
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print("\n📌 Confusion Matrix:\n", cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    # Feature importance
    feature_importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    print("\n🔥 Feature Importances:\n", feature_importances)

    # Plot feature importance
    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="viridis")
    plt.title("Feature Importance from Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()


if __name__ == "__main__":
    main()
