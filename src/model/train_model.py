from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class AdvancedStockPredictor:
    """
    Advanced stock prediction model using ensemble methods and multiple algorithms.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        self.best_model = None
        self.best_score = 0
        self.n_features = None

    def prepare_features(self, X):
        """
        Prepare features by scaling and adding technical indicators.
        """
        try:
            # Ensure X is 2D
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            
            # Scale the features
            X_scaled = self.scaler.fit_transform(X)
            
            # Convert to DataFrame for easier manipulation
            X_df = pd.DataFrame(X_scaled)
            
            # Store the original number of features
            self.n_features = X_df.shape[1]
            
            return X_scaled  # Return only scaled features without adding moving averages
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return X

    def train_model(self, X, y):
        """
        Train multiple models and select the best performing one.
        """
        # Prepare features
        X_prepared = self.prepare_features(X)
        self.n_features = X_prepared.shape[1]  # Store number of features
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_prepared, y, test_size=0.2, random_state=42
        )

        # Train and evaluate each model
        results = {}
        for name, model in self.models.items():
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                predictions = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(y_test, predictions)
                recall = recall_score(y_test, predictions)
                
                # Perform cross-validation
                cv_scores = cross_val_score(model, X_prepared, y, cv=5)
                
                # Store results
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'model': model
                }
                
                print(f"\n{name.upper()} Results:")
                print(f"Accuracy: {accuracy:.3f}")
                print(f"Precision: {precision:.3f}")
                print(f"Recall: {recall:.3f}")
                print(f"Cross-validation Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue

        if not results:
            raise ValueError("No models were successfully trained")

        # Select the best model based on cross-validation score
        best_model_name = max(results.items(), key=lambda x: x[1]['cv_mean'])[0]
        self.best_model = results[best_model_name]['model']
        self.best_score = results[best_model_name]['cv_mean']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Cross-validation Score: {self.best_score:.3f}")
        
        return self.best_model

    def predict(self, X):
        """
        Make predictions using the best trained model.
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Ensure X has the same number of features as training data
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Scale the features
        X_prepared = self.prepare_features(X)
        return self.best_model.predict(X_prepared)

    def predict_proba(self, X):
        """
        Get probability predictions using the best trained model.
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Ensure X has the same number of features as training data
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        X_prepared = self.prepare_features(X)
        return self.best_model.predict_proba(X_prepared)

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)  # 5 features
    y = (X.sum(axis=1) > 0).astype(int)  # Binary classification
    
    # Create and train the model
    predictor = AdvancedStockPredictor()
    best_model = predictor.train_model(X, y)
    
    # Make predictions
    sample_data = np.random.randn(5, 5)
    predictions = predictor.predict(sample_data)
    probabilities = predictor.predict_proba(sample_data)
    
    print("\nSample Predictions:")
    print(predictions)
    print("\nPrediction Probabilities:")
    print(probabilities) 