import pandas as pd

from datasets import Dataset, DatasetDict
from src.utils import (
    DatasetNormalizer,
    FileHandler,
    group_datasets,
    merge_datasets,
)
from src.utils.logging import get_logger

_logger = get_logger(__name__)


def normalize_tass2020_sentiment(configs: dict) -> DatasetDict:
    """
    Normalize TASS 2020 sentiment datasets based on the provided configurations.

    Args:
        configs (dict): Dictionary containing dataset and normalization configurations.

    Returns:
        tuple: A tuple containing the normalized training and testing datasets.

    Raises:
        Exception: If any error occurs during normalization.
    """
    _logger.info("Starting TASS 2020 sentiment normalization process.")
    try:
        # Use the filehandler to process the input files
        train_file_handler = FileHandler(configs["dataset"]["train_files"])
        test_file_handler = FileHandler(configs["dataset"]["test_files"])

        train_datasets = train_file_handler.process_files()
        test_datasets = test_file_handler.process_files()
        normalizer = DatasetNormalizer(configs)

        # TEST DS
        # group datasets
        test_grouped_datasets = group_datasets(test_datasets)

        # merge & concat datasets
        test_merged_ds = merge_datasets(
            config=configs, dfs=test_grouped_datasets, merge_col="ID"
        )
        test_unnorm_ds = pd.concat(test_merged_ds, ignore_index=True)
        test_norm_ds = normalizer.standard_cleanup(test_unnorm_ds)

        # TRAIN DS
        train_datasets_v = []
        for ds in train_datasets.values():
            ds.columns = ["id", "texts", "labels"]
            train_datasets_v.append(ds)

        train_unnorm_ds = normalizer.add_language_variation_column(
            train_datasets_v, configs["dataset"]["train_files"]
        )

        train_unnorm_ds = pd.concat(train_datasets.values(), ignore_index=True)
        train_norm_ds = normalizer.standard_cleanup(train_unnorm_ds)

        _logger.info(
            "TASS 2020 sentiment normalization process completed successfully."
        )

        dataset_dict_list = []

        # include full dataset wo lng variation
        train_hf_ds_all = train_norm_ds.drop(columns=["language_variation"])
        train_hf_ds_all = Dataset.from_pandas(train_hf_ds_all)

        test_hf_ds_all = test_norm_ds.drop(columns=["language_variation"])
        test_hf_ds_all = Dataset.from_pandas(test_hf_ds_all)

        # include lng variation
        train_hf_ds = Dataset.from_pandas(train_norm_ds)
        test_hf_ds = Dataset.from_pandas(test_norm_ds)

        all_languages = set(test_hf_ds["language_variation"])

        for language in all_languages:
            _train_hf_ds = train_hf_ds.filter(
                lambda x: x["language_variation"] == language
            )
            _test_hf_ds = test_hf_ds.filter(
                lambda x: x["language_variation"] == language
            )
            dataset_dict = DatasetDict(
                {"train": _train_hf_ds, "test": _test_hf_ds}
            )

            dataset_dict_list.append(dataset_dict)
        return dataset_dict_list
    except Exception as e:
        _logger.error(f"Error in normalize_tass2020_sentiment: {e}")
        raise
