import pandas as pd

from datasets import Dataset, DatasetDict
from src.utils.dataset_normalizer import DatasetNormalizer
from src.utils.filehandler import FileHandler
from src.utils.logging import get_logger

_logger = get_logger(__name__)


def clean_vaxxstance(configs: dict) -> DatasetDict:
    """
    Clean and normalize the VaxxStance dataset based on the provided configurations.

    Args:
        configs (dict): Dictionary containing dataset and normalization configurations.

    Returns:
        tuple: A tuple containing the normalized training and testing datasets.

    Raises:
        Exception: If any error occurs during normalization.
    """
    _logger.info("Starting VaxxStance dataset cleaning process.")
    try:
        train_file_handler = FileHandler(configs["dataset"]["train_files"])
        test_file_handler = FileHandler(configs["dataset"]["test_files"])

        train_datasets = train_file_handler.process_files()
        test_datasets = test_file_handler.process_files()

        normalizer = DatasetNormalizer(configs)

        # Further processing of datasets if needed
        train_datasets = normalizer.add_language_variation_column(
            train_datasets.values(), configs["dataset"]["train_files"]
        )
        test_datasets = normalizer.add_language_variation_column(
            test_datasets.values(), configs["dataset"]["test_files"]
        )

        # concatenate
        train_unnorm_ds = pd.concat(train_datasets, ignore_index=True)
        test_unnorm_ds = pd.concat(test_datasets, ignore_index=True)

        # normalize
        train_norm_ds = normalizer.standard_cleanup(train_unnorm_ds)
        train_norm_ds.reset_index(drop=True, inplace=True)

        test_norm_ds = normalizer.standard_cleanup(test_unnorm_ds)
        test_norm_ds.reset_index(drop=True, inplace=True)

        # grab the wanted language

        _logger.info(
            "VaxxStance dataset cleaning process completed successfully."
        )

        train_hf_ds = Dataset.from_pandas(train_norm_ds)
        test_hf_ds = Dataset.from_pandas(test_norm_ds)
        dataset_dict = DatasetDict({"train": train_hf_ds, "test": test_hf_ds})

        return [dataset_dict]

    except Exception as e:
        _logger.error(f"Error in clean_vaxxstance: {e}")
        raise
