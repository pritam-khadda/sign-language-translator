import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
data = pd.read_csv("data.csv", header=None)

# Features and Labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Dataset information
print("=" * 50)
print("Dataset Shape:", data.shape)
print("Feature Shape:", X.shape)
print("Label Shape:", y.shape)
print("\nSamples per Gesture:")
print(y.value_counts())
print("=" * 50)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Save Model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")