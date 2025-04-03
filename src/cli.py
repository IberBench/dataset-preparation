import os
import time
from pathlib import Path

import typer
from huggingface_hub import HfApi, create_repo

from src.ds_preprocessing.cleaning_fn import cleaning_registry
from src.models.config import Config
from src.utils import (
    add_to_main_dataset,
    append_to_hf_file,
    auth_check,
    create_dataset_metadata,
    create_dataset_name,
    create_repo_name,
    dataset_results_path,
    find_files_with_suffix,
    generate_dataset_card_from_urls,
    load_configs,
    populate_template,
    save_json,
    upload_dataset,
    upload_hf_file,
)
from src.utils.logging import get_logger

_logger = get_logger(__name__)

app = typer.Typer()
app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def normalizer_and_save(
    config_path: Path = "configs/to_hf/test/",
    root_path: Path = Path("results"),
):
    """
    Normalizes and saves a dataset based on the provided configuration.

    Args:
        config_path (Path): Path to the configuration file.
        results_path (Path): Path to save the cleaned dataset.

    This function performs the following steps:
    1. Loads the configuration from the specified path.
    2. Cleans the dataset based on the configuration.
    3. Saves the cleaned dataset to the specified results path.
    """

    for file in config_path.iterdir():
        # load config
        _logger.info(f"Loading config from {file}")
        config: Config = load_configs(file)

        try:
            # normalize dataset
            _logger.info("Cleaning dataset")
            normalize_custom_fn = cleaning_registry[
                config.normalizer.normalizer_fn
            ]
            normalized_ds = normalize_custom_fn(config.model_dump())
        except KeyError as e:
            _logger.error(f"Cleaning function not found in registry: {e}")
            raise
        for ds in normalized_ds:
            # Save cleaned huggingface dataset in the results_path
            results_path = dataset_results_path(
                root_path,
                task_config=config.task,
                dataset=ds,
            )
            _logger.info(f"Saving normalized dataset to {results_path}")
            ds.save_to_disk(results_path)

            # Save task metadata in results_path
            metadata = create_dataset_metadata(config, ds)
            save_json(results_path / "task_metadata.json", metadata)

            _logger.info("Dataset saved successfully")


@app.command()
def upload_ds_to_huggingface(
    config_path: Path = "configs/to_hf/test/",
    main_hf_dataset: str = "iberbench/iberbench_all",
    root_path: Path = Path("results"),
    add_to_main_ds: bool = True,
):
    """
    Upload datasets to Hugging Face Hub.

    Args:
        config_path (Path): Path to the configuration directory.
        main_hf_dataset (str): Main Hugging Face dataset repository name.
        dataset_path (Path): Path to the dataset results directory.
        add_to_main_ds (bool): Flag to add the dataset to the main dataset repository.
    """
    for file in config_path.iterdir():
        config: Config = load_configs(file)
        dataset_name = create_dataset_name(config.task)
        matching_files = find_files_with_suffix(root_path, dataset_name)
        for file_name in matching_files:
            repo_name = create_repo_name(
                prefix="iberbench", dataset_name=file_name
            )
            results_path = root_path / file_name

            # Create the repository for the dataset
            if not auth_check(repo_name):
                create_repo(
                    repo_name, repo_type="dataset", private=True, token=True
                )
                time.sleep(20)

            # Upload the dataset
            upload_dataset(config.model_dump(), repo_name, results_path)

            # Upload to the main dataset
            if add_to_main_ds:
                add_to_main_dataset(repo_name, results_path, main_hf_dataset)

            # Upload metadata
            upload_hf_file(
                path_or_fileobj=results_path / "task_metadata.json",
                path_in_repo="task_metadata.json",
                repo_id=repo_name,
                token=os.environ["HF_API_KEY"],
            )


@app.command()
def create_model_card(config_path: Path = Path("configs/to_hf/test/")):
    for file in config_path.iterdir():
        _logger.info(f"Processing configuration file: {file}")
        config: Config = load_configs(file)

        # Generate dataset card from all URLs
        combined_dataset_card = generate_dataset_card_from_urls(config.task.url)
        data_fields_json = combined_dataset_card.model_dump()

        # Create the markdown output using the template
        markdown_output = populate_template(combined_dataset_card)

        # update files in hf repo
        files_to_update = {
            "task_metadata.json": data_fields_json,
            "README.md": markdown_output,
        }

        task_path = create_dataset_name(config.task)

        for file_name, content in files_to_update.items():
            append_to_hf_file(
                file_name=file_name,
                repo_id=f"iberbench/{task_path}",
                token=os.environ["HF_API_KEY"],
                new_content=content,
                save_path=f"results/{task_path}",
            )


@app.command()
def upload_to_hf(path_to_upload: Path, repo_name: str):
    """
    Upload all the folders and files from `path_to_upload` to the `repo_name`
    repository in HuggingFace's hub, excluding those related with the dataset:
    namely split folders and dataset_dict.json.

    Args:
        path_to_upload (Path): Path to be uploaded to the hub.
        repo_name (str): name of the repository where to push the path content
    """
    client = HfApi()
    _logger.info("Starting the upload process to Hugging Face Hub.")
    for file in path_to_upload.glob("*"):
        if file.is_dir() and file.name not in {"train", "validation", "test"}:
            client.upload_folder(
                folder_path=file,
                path_in_repo=file.name,
                repo_id=repo_name,
                token=os.environ["HF_API_KEY"],
                repo_type="dataset",
            )
        if file.is_file() and file.name != "dataset_dict.json":
            client.upload_file(
                path_or_fileobj=file,
                path_in_repo=file.name,
                repo_id=repo_name,
                token=os.environ["HF_API_KEY"],
                repo_type="dataset",
            )
    _logger.info("Upload process completed.")


if __name__ == "__main__":
    app()
