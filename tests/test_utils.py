import pytest
from utils import encode_binary_features


import pandas as pd


@pytest.fixture
def sample_dataframe():
    """Fixture to provide a sample DataFrame for testing."""
    data = {
        "feature_a": ["yes", "no", "yes", "no"],
        "feature_b": ["no", "yes", "no", "yes"],
        "feature_c": [1, 2, 3, 4],
    }
    return pd.DataFrame(data)


def test_encode_binary_features_success(sample_dataframe):
    """Test successful encoding of binary features."""
    df = sample_dataframe.copy()
    binary_features = ["feature_a", "feature_b"]
    encode_binary_features(df, binary_features)

    pd.testing.assert_series_equal(
        df["feature_a"], pd.Series([1, 0, 1, 0], name="feature_a")
    )
    pd.testing.assert_series_equal(
        df["feature_b"], pd.Series([0, 1, 0, 1], name="feature_b")
    )


def test_encode_binary_features_key_error(sample_dataframe, capsys):
    """Test handling of KeyError when a binary feature is not found."""
    df = sample_dataframe.copy()
    binary_features = ["feature_a", "non_existent_feature"]
    encode_binary_features(df, binary_features)
    captured = capsys.readouterr()
    assert (
        "An unexpected key error occurred during binary encoding: 'non_existent_feature'"
        in captured.out
    )


def test_encode_binary_features_empty_list(sample_dataframe):
    """Test with an empty list of binary features."""
    df = sample_dataframe.copy()
    binary_features = []
    encode_binary_features(df, binary_features)
    # Ensure the DataFrame remains unchanged
    pd.testing.assert_frame_equal(df, sample_dataframe)


def test_encode_binary_features_no_binary_columns(sample_dataframe):
    """Test with binary features that are not 'yes'/'no'."""
    df = sample_dataframe.copy()
    binary_features = ["feature_c"]
    encode_binary_features(df, binary_features)
    # The .map() operation will introduce NaNs for values not in the mapping
    pd.testing.assert_series_equal(
        df["feature_c"],
        pd.Series(
            [float("nan"), float("nan"), float("nan"), float("nan")], name="feature_c"
        ),
    )
