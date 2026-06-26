import joblib

from sklearn.ensemble import RandomForestClassifier


class Trainer:

    def train(self, X_train, y_train):

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        return model

    def save_model(self, model):

        joblib.dump(
            model,
            "models/modelo.pkl"
        )