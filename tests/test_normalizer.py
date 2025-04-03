# filepath: /data/research/users/iborrego/code/ivace_datasets/tests/test_normalizer.py
import unittest
from unittest.mock import patch, MagicMock
from datasets import Dataset, DatasetDict
from src.ds_preprocessing.normalizer import ds_normalizer
from src.models.config import Config


class TestDsNormalizer(unittest.TestCase):

    @patch("src.ds_preprocessing.normalizer.cleaning_registry")
    @patch("src.ds_preprocessing.normalizer.Dataset")
    def test_ds_normalizer(self, mock_dataset, mock_cleaning_registry):
        # Mock the cleaning function
        mock_cleaning_fn = MagicMock()
        mock_cleaning_fn.return_value = (MagicMock(), MagicMock())
        mock_cleaning_registry.__getitem__.return_value = mock_cleaning_fn

        # Mock the Dataset.from_pandas method
        mock_dataset.from_pandas.side_effect = [MagicMock(), MagicMock()]

        # Create a mock config
        config = MagicMock()
        config.normalizer.normalizer_fn = "mock_fn"

        # Call the ds_normalizer function
        result = ds_normalizer(config)

        # Assertions
        self.assertIsInstance(result, DatasetDict)
        self.assertIn("train", result)
        self.assertIn("test", result)
        mock_cleaning_registry.__getitem__.assert_called_once_with("mock_fn")
        mock_cleaning_fn.assert_called_once_with(config)
        self.assertEqual(mock_dataset.from_pandas.call_count, 2)


if __name__ == "__main__":
    unittest.main()
