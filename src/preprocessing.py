import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def drop_columns(train_df, test_df):
    columns_to_drop = [
        "Unnamed: 0",
        "trans_num",
        "first",
        "last",
        "street"
    ]

    train_df = train_df.drop(columns=columns_to_drop)
    test_df = test_df.drop(columns=columns_to_drop)

    return train_df, test_df


def process_date_columns(train_df, test_df):
    """
    Convert date columns into useful numerical features.
    """

    train_df["trans_date_trans_time"] = pd.to_datetime(train_df["trans_date_trans_time"])
    test_df["trans_date_trans_time"] = pd.to_datetime(test_df["trans_date_trans_time"])

    train_df["transaction_hour"] = train_df["trans_date_trans_time"].dt.hour
    test_df["transaction_hour"] = test_df["trans_date_trans_time"].dt.hour

    train_df["transaction_day"] = train_df["trans_date_trans_time"].dt.day
    test_df["transaction_day"] = test_df["trans_date_trans_time"].dt.day

    train_df["transaction_month"] = train_df["trans_date_trans_time"].dt.month
    test_df["transaction_month"] = test_df["trans_date_trans_time"].dt.month

    train_df["dob"] = pd.to_datetime(train_df["dob"])
    test_df["dob"] = pd.to_datetime(test_df["dob"])

    train_df["age"] = (
        train_df["trans_date_trans_time"].dt.year
        - train_df["dob"].dt.year
    )

    test_df["age"] = (
        test_df["trans_date_trans_time"].dt.year
        - test_df["dob"].dt.year
    )

    train_df = train_df.drop(columns=["trans_date_trans_time", "dob"])
    test_df = test_df.drop(columns=["trans_date_trans_time", "dob"])

    return train_df, test_df


def encode_features(train_df, test_df):
    """
    Encode categorical columns using OrdinalEncoder.
    """

    categorical_columns = train_df.select_dtypes(include=["object"]).columns

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    train_df[categorical_columns] = encoder.fit_transform(train_df[categorical_columns])
    test_df[categorical_columns] = encoder.transform(test_df[categorical_columns])

    return train_df, test_df