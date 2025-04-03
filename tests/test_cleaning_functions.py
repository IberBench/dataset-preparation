import unittest
from unittest.mock import patch, MagicMock
from src.ds_preprocessing.cleaning_fn.classification_norm import (
    standard_classification_normalizer,
)
from src.ds_preprocessing.cleaning_fn.tass2020_sentiment import (
    normalize_tass2020_sentiment,
)
from src.ds_preprocessing.cleaning_fn.vaxxstance_2020 import clean_vaxxstance


class TestCleaningFunctions(unittest.TestCase):

    @patch("src.ds_preprocessing.cleaning_fn.classification_norm.DatasetNormalizer")
    @patch("src.ds_preprocessing.cleaning_fn.classification_norm.FileHandler")
    def test_standard_classification_normalizer(
        self, mock_file_handler, mock_dataset_normalizer
    ):
        # Mock the FileHandler
        mock_train_file_handler = MagicMock()
        mock_test_file_handler = MagicMock()
        mock_file_handler.side_effect = [
            mock_train_file_handler,
            mock_test_file_handler,
        ]

        # Mock the process_files method
        mock_train_file_handler.process_files.return_value = {"train": MagicMock()}
        mock_test_file_handler.process_files.return_value = {"test": MagicMock()}

        # Mock the DatasetNormalizer
        mock_normalizer = mock_dataset_normalizer.return_value
        mock_normalizer.standard_cleanup.side_effect = [MagicMock(), MagicMock()]

        # Create a mock config
        configs = {
            "dataset": {
                "train_files": "/data/research/users/iborrego/code/ivace_datasets/datasets/vaxxstance_2021/dataset/train",
                "test_files": "/data/research/users/iborrego/code/ivace_datasets/datasets/vaxxstance_2021/dataset/test",
            },
            "normalizer": {"columns": ["col1", "col2"]},
        }

        # Call the standard_classification_normalizer function
        train_norm_ds, test_norm_ds = standard_classification_normalizer(configs)

        # Assertions
        self.assertIsNotNone(train_norm_ds)
        self.assertIsNotNone(test_norm_ds)
        mock_file_handler.assert_any_call(
            "/data/research/users/iborrego/code/ivace_datasets/datasets/vaxxstance_2021/dataset/train"
        )
        mock_file_handler.assert_any_call(
            "/data/research/users/iborrego/code/ivace_datasets/datasets/vaxxstance_2021/dataset/test"
        )
        mock_train_file_handler.process_files.assert_called_once()
        mock_test_file_handler.process_files.assert_called_once()
        mock_normalizer.standard_cleanup.assert_any_call(
            mock_train_file_handler.process_files.return_value["train"]
        )
        mock_normalizer.standard_cleanup.assert_any_call(
            mock_test_file_handler.process_files.return_value["test"]
        )

    @patch("src.ds_preprocessing.cleaning_fn.tass2020_sentiment.pd.concat")
    def test_normalize_tass2020_sentiment(self, mock_concat):
        # Mock the return value of pd.concat
        mock_concat.return_value = MagicMock()

        # Create a mock config
        configs = {
            "dataset": {
                "train_files": "/data/research/users/iborrego/code/ivace_datasets/datasets/vaxxstance_2021/dataset/train",
                "test_files": "/data/research/users/iborrego/code/ivace_datasets/datasets/vaxxstance_2021/dataset/test",
            },
            "normalizer": {"language": "spanish"},
            "columns": ["col1", "col2"],
        }

        # Call the function with the necessary arguments
        test_norm_ds, train_norm_ds = normalize_tass2020_sentiment(configs)

        # Add assertions to verify the expected behavior
        self.assertIsNotNone(test_norm_ds)
        self.assertIsNotNone(train_norm_ds)

    @patch("src.ds_preprocessing.cleaning_fn.vaxxstance_2020.FileHandler")
    def test_clean_vaxxstance(self, mock_file_handler):
        # Mock the FileHandler
        mock_file_handler.return_value = MagicMock()

        # Create a mock config
        configs = {
            "dataset": {
                "train_files": "/data/research/users/iborrego/code/ivace_datasets/datasets/vaxxstance_2021/dataset",
                "test_files": "/data/research/users/iborrego/code/ivace_datasets/datasets/vaxxstance_2021/dataset/test",
            },
            "normalizer": {
                "columns": ["col1", "col2"],
                "language": "spanish",
                "input_cols": ["text"],
                "output_col": ["text"],
                "keep_columns": ["col1", "col2"],
            },
            "mapping": {"language_variation": {"es": "spain", "eu": "basque country"}},
        }

        # Call the function with the necessary arguments
        train_norm_ds, test_norm_ds = clean_vaxxstance(configs)

        # Add assertions to verify the expected behavior
        self.assertIsNotNone(train_norm_ds)
        self.assertIsNotNone(test_norm_ds)
        mock_file_handler.assert_any_call(
            "/data/research/users/iborrego/code/ivace_datasets/datasets/vaxxstance_2021/dataset"
        )
