from sklearn.metrics import classification_report


class Evaluator:

    def evaluate(self, model, X_test, y_test):

        y_pred = model.predict(X_test)

        print(
            classification_report(
                y_test,
                y_pred
            )
        )