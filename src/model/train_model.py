from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    return model

# Example usage
if __name__ == "__main__":
    # Dummy data
    X = [[0.1], [0.4], [0.5], [0.7], [0.9]]
    y = [0, 0, 1, 1, 1]
    train_model(X, y) 