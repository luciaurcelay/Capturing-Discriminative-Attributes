from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class SVC_Trainer:
    def __init__(self, seed, kernel, C, gamma):
        self.seed = seed
        self.classifier = SVC(random_state=seed, kernel=kernel, C=C, gamma=gamma)

    def search_best_params(self, X_train, y_train):
        # Define the parameter values to search
        param_grid = {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10]}

        # Create a grid search object
        grid_search = GridSearchCV(self.classifier, param_grid, cv=5, scoring="f1")

        # Fit the grid search object to the data
        grid_search.fit(X_train, y_train)

        # Get the best parameters
        best_params = grid_search.best_params_

        return best_params

    def train_classifier(self, X_train, y_train, X_val):
        self.classifier.fit(X_train, y_train)
        predictions = self.classifier.predict(X_val)

        return predictions

    def save_model(self, save_path):
        pass

    def save_features(self, save_path):
        pass
