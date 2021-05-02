import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from pathlib import Path
import pickle


def clean_df():
    DATA_DIR = Path("data")
    df = pd.read_csv(DATA_DIR / Path("Parkinsons - Copy of Sheet1.csv"))
    df.head()

    # replace healthy with 0 and parkinsons with 1
    df["sourceName"] = [
        1 if item == "Parkinsons" else 0 for item in list(df["sourceName"])
    ]

    # convert to one hot if necessary
    # df = pd.get_dummies(df)
    df.head()

    # remove target from df and move into its own list
    labels = np.array(df["sourceName"])
    feature_list = [i for i in list(df.columns) if i != "sourceName"]
    df.drop(["sourceName"], axis=1, inplace=True)
    df.head()

    pd_df = df  # store pandas version
    df = np.array(df)

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        df, labels, test_size=0.25, random_state=42
    )

    return (
        train_features,
        test_features,
        train_labels,
        test_labels,
        df,
        pd_df,
        labels,
        feature_list,
    )


def train_model(train_features, train_labels, test_features, test_lables):
    # Instantiate model
    n_estimators = 1000
    random_state = 42
    reg_model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state
    )
    class_model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )

    # Train the models on training data
    reg_model.fit(train_features, train_labels)
    class_model.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    reg_pred = reg_model.predict(test_features)
    class_pred = class_model.predict(test_features)

    return reg_model, reg_pred, class_model, class_pred


def show_pred_real(test_features, test_labels, predictions):
    """
    Creates a new df with both real and predicted sourceName
    for the test dataset.
    Includes error for predicted to real.
    Correct is true if predicted is within 0.5 of real
    """
    combined_df = pd.DataFrame(test_features, columns=["double", "speed", "length"])
    combined_df["real_sourceName"] = test_labels
    combined_df["pred_sourceName"] = predictions

    combined_df["correct"] = (
        combined_df["real_sourceName"] == combined_df["pred_sourceName"]
    )
    combined_df.head(10)

    return combined_df


def show_metrics(
    reg_pred,
    class_pred,
    test_labels,
    feature_list,
    test_features,
    reg_model,
    class_model,
):
    def reg_metrics(predictions):
        # Calculate the absolute errors
        errors = abs(predictions - test_labels)
        print("Regression mean error:", round(np.mean(errors), 2))

    def class_metrics(predictions):
        combined_df = show_pred_real(test_features, test_labels, predictions)
        correct_dist = combined_df["correct"].value_counts().to_dict()
        accuracy = correct_dist[True] / (correct_dist[True] + correct_dist[False])
        print("Classification accuracy:", accuracy)

    def get_importances(model, model_name="model"):
        # Get numerical feature importances
        importances = list(model.feature_importances_)

        # List of tuples with variable and importance
        feature_importances = [
            (feature, round(importance, 2))
            for feature, importance in zip(feature_list, importances)
        ]

        # Sort the feature importances by most important first
        feature_importances = sorted(
            feature_importances, key=lambda x: x[1], reverse=True
        )

        # Print out the feature and importances
        print(f"Feature importances for {model_name}: ")
        [
            print("Variable: {:20} Importance: {}".format(*pair))
            for pair in feature_importances
        ]

    reg_metrics(reg_pred)
    class_metrics(class_pred)
    print()
    get_importances(reg_model, "regression")
    get_importances(class_model, "classification")


def save_models(reg_model, class_model):
    if input("save? y/n ").lower().strip() == "y":
        pickle.dump(reg_model, open(Path("models/reg_model.sav"), "wb"))
        pickle.dump(class_model, open(Path("models/class_model.sav"), "wb"))
        print("saved 2 models")
    else:
        print("not saved")


if __name__ == "__main__":

    (
        train_features,
        test_features,
        train_labels,
        test_labels,
        df,
        pd_df,
        labels,
        feature_list,
    ) = clean_df()

    reg_model, reg_pred, class_model, class_pred = train_model(
        train_features, train_labels, test_features, test_labels
    )

    show_metrics(
        reg_pred,
        class_pred,
        test_labels,
        feature_list,
        test_features,
        reg_model,
        class_model,
    )

    save_models(reg_model, class_model)
