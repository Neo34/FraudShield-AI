from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class Preprocessor:

    def split_data(self, X, y):

        return train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y
        )

    def apply_smote(self, X_train, y_train):

        smote = SMOTE(random_state=42)

        return smote.fit_resample(
            X_train,
            y_train
        )