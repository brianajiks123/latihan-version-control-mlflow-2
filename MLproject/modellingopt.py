import mlflow, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001/")  # 5000 is already busy

# Create a new MLflow Experiment
mlflow.set_experiment("Latihan Credit Scoring")

# Load the data with error handling
try:
    data = pd.read_csv("train_pca.csv")
    
    print(f"Data loaded successfully with shape {data.shape}")
except FileNotFoundError:
    print("Error: The file 'train_pca.csv' was not found.")
    
    exit(1)

# Handle missing values (if any)
if data.isnull().sum().any():
    print("Missing values detected. Filling missing values with median.")
    
    data.fillna(data.median(), inplace=True)

# Ensure correct data types: Convert int to float
data = data.apply(lambda col: col.astype(float) if col.dtype == 'int64' else col)

# Check for target column
if "Credit_Score" not in data.columns:
    print("Error: 'Credit_Score' column not found in the dataset.")
    
    exit(1)

# Split data into features and target
X = data.drop("Credit_Score", axis=1)
y = data["Credit_Score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

input_example = X_train.iloc[0:5]

# Log parameters and models
with mlflow.start_run():
    try:
        # Define hyperparameters
        n_estimators = 505
        max_depth = 37
        
        # Enable auto logging
        mlflow.autolog()

        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log model and metrics
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"Initial Model Accuracy: {accuracy:.4f}")
    
    except Exception as e:
        print(f"Error during model training or logging: {e}")
        
        exit(1)

# Hyperparameter tuning with validation
n_estimators_range = np.linspace(10, 1000, 5, dtype=int)
max_depth_range = np.linspace(1, 50, 5, dtype=int)

best_accuracy = 0
best_params = {}

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"elastic_search_{n_estimators}_{max_depth}"):
            try:
                mlflow.autolog()

                # Train model
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)

                # Evaluate model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Log metrics and model
                mlflow.log_metric("accuracy", accuracy)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                    
                    print(f"New Best Model with Accuracy: {accuracy:.4f}")
                    
                    # Log the best model
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        input_example=input_example
                    )

            except Exception as e:
                print(f"Error during hyperparameter tuning with n_estimators={n_estimators}, max_depth={max_depth}: {e}")

# Output the best parameters found
if best_params:
    print(f"Best model parameters: {best_params} with accuracy: {best_accuracy:.4f}")
else:
    print("No improvement in model accuracy during hyperparameter search.")
