from sklearn.svm import SVC

class SVC_Trainer:

    def __init__(self, seed, kernel, C, gamma):
        self.seed = seed
        self.classifier = SVC(random_state=seed, kernel=kernel, C=C, gamma=gamma)

    def train_classifier(self, X_train, y_train, X_val):
        self.classifier.fit(X_train, y_train)
        predictions = self.classifier.predict(X_val)

        return predictions
    
    def save_model(self, save_path):
        pass


    def save_features(self, save_path):
        pass