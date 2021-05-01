import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def clean_df():
    df = pd.read_csv("DadDoubleSupport - AiDadMom-2.csv")
    df.head()

    # convert to one hot if necessary
    df = pd.get_dummies(df)
    df.head()

    # what to predict
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
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)

    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)

    return rf, predictions


def show_metrics(predictions, test_labels, feature_list):
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)

    print("Mean error:", round(np.mean(errors), 2))

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)

    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print("Accuracy:", round(accuracy, 2), "%.")

    # Get numerical feature importances
    importances = list(rf.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [
        (feature, round(importance, 2))
        for feature, importance in zip(feature_list, importances)
    ]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    # Print out the feature and importances
    [
        print("Variable: {:20} Importance: {}".format(*pair))
        for pair in feature_importances
    ]
    return errors


def show_pred_real(test_features, test_labels, predictions):
    """
    Creates a new df with both real and predicted sourceName
    for the test dataset.
    Includes error for predicted to real.
    Correct is true if predicted is within 0.5 of real
    """
    combined_df = pd.DataFrame(
        test_features, columns=["double", "speed", "length", "Asym"]
    )
    combined_df["real_sourceName"] = test_labels
    combined_df["pred_sourceName"] = predictions
    combined_df["error"] = (
        combined_df["pred_sourceName"] - combined_df["real_sourceName"]
    ) / combined_df["real_sourceName"]
    combined_df["correct"] = (
        abs(combined_df["error"]) * combined_df["real_sourceName"]
    ) < 0.5
    combined_df.head(10)

    return combined_df


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

    rf, predictions = train_model(
        train_features, train_labels, test_features, test_labels
    )

    errors = show_metrics(predictions, test_labels, feature_list)
