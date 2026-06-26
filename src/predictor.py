import joblib


class Predictor:

    def __init__(self):

        self.model = joblib.load(
            "models/modelo.pkl"
        )

    def predict(self, transaction):

        return self.model.predict(
            transaction
        )