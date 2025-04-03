from datasets import Dataset, DatasetDict, load_dataset
from src.utils.dataset_normalizer import DatasetNormalizer
from src.utils.logging import get_logger

_logger = get_logger(__name__)


def hf_repo_normalizer(configs: dict):
    _logger.info("Starting hf repo normalization process.")
    try:
        train_dataset = load_dataset(
            path=configs["dataset"]["hf_repo_id"],
            name=configs["dataset"]["hf_subset"],
            split="train",
            trust_remote_code=True,
        )
        train_df = train_dataset.to_pandas()

        test_dataset = load_dataset(
            path=configs["dataset"]["hf_repo_id"],
            name=configs["dataset"]["hf_subset"],
            split="test",
            trust_remote_code=True,
        )
        test_df = test_dataset.to_pandas()

        normalizer = DatasetNormalizer(configs)

        train_norm_ds = normalizer.standard_cleanup(train_df)
        test_norm_ds = normalizer.standard_cleanup(test_df)

        _logger.info(
            "Standard classification normalization process completed successfully."
        )

        # select right language
        if len(train_norm_ds.language.unique()) > 1:
            train_norm_ds = train_norm_ds[
                train_norm_ds["language"] == configs["task"]["language"]
            ]
            train_norm_ds.reset_index(drop=True, inplace=True)

            test_norm_ds = test_norm_ds[
                test_norm_ds["language"] == configs["task"]["language"]
            ]
            test_norm_ds.reset_index(drop=True, inplace=True)

        train_hf_ds = Dataset.from_pandas(train_norm_ds)
        test_hf_ds = Dataset.from_pandas(test_norm_ds)
        dataset_dict = DatasetDict({"train": train_hf_ds, "test": test_hf_ds})

        return [dataset_dict]

    except Exception as e:
        _logger.error(f"Error in standard_classification_normalizer: {e}")
        raise
