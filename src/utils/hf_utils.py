import os
import time
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, create_repo, hf_hub_download

from datasets import DatasetDict, load_dataset, load_from_disk
from src.utils.logging import get_logger

_logger = get_logger(__name__)


def download_hf_file(
    file_name: str, repo_id: str, token: str, save_path: str
) -> str:
    """
    Download a file from a Hugging Face repository.

    Args:
        file_name (str): The name of the file to download.
        repo_id (str): The ID of the Hugging Face repository.
        token (str): The Hugging Face API token.
        save_path (str): The local path to save the file.

    Returns:
        str: The content of the downloaded file.
    """
    _logger.info(f"Downloading {file_name} from repo: {repo_id}")

    # Download the specified file
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_name,
        repo_type="dataset",  # Change to "model" if it's a model repo
        token=token,
    )

    # Ensure the local directory exists
    save_path = Path(save_path) / file_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the file content to the specified path
    with open(save_path, "w") as file:
        with open(file_path, "r") as downloaded_file:
            content = downloaded_file.read()
            file.write(content)

    _logger.info(f"{file_name} downloaded and saved to: {save_path}")
    return content


def upload_hf_file(
    path_or_fileobj, path_in_repo, repo_id, token, repo_type="dataset"
):
    client = HfApi()
    client.upload_file(
        path_or_fileobj=path_or_fileobj,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        token=token,
        repo_type=repo_type,
    )


def append_to_hf_file(
    file_name: str,
    repo_id: str,
    token: str,
    new_content,
    save_path: str,
    separator: str = "\n\n<!-- New content added on {date} -->\n\n",
) -> None:
    """
    Appends new content to an existing file in a Hugging Face repository.
    If the file doesn't exist, it creates it with just the new content.
    For JSON files, merges the dictionaries instead of appending.

    Args:
        file_name: Name of the file to update
        repo_id: Hugging Face repository ID
        token: Authentication token
        new_content: Content to append/merge (string for text, dict for JSON)
        save_path: Local path to save the file temporarily
        separator: Separator for text files (ignored for JSON)
    """
    _logger.info(f"Appending to file {file_name} in repository {repo_id}")

    # Check if we're dealing with a JSON file and dictionary content
    is_json = file_name.endswith(".json") and isinstance(new_content, dict)

    try:
        if is_json:
            # For JSON files, try to load existing JSON and merge
            try:
                # Download the file
                file_path = download_hf_file(
                    file_name=file_name,
                    repo_id=repo_id,
                    token=token,
                    save_path=save_path,
                )

                # Parse the existing JSON
                import json

                with open(Path(save_path) / file_name, "r") as f:
                    existing_json = json.load(f)

                _logger.info(
                    f"Successfully loaded existing JSON from {file_name}"
                )

                # Merge dictionaries (existing is updated with new)
                combined_content = {**existing_json, **new_content}
                _logger.info(
                    f"Merged JSON with {len(existing_json)} existing and {len(new_content)} new fields"
                )
            except Exception as e:
                _logger.warning(
                    f"Could not load or merge existing JSON: {e}. Using new JSON only."
                )
                combined_content = new_content
        else:
            # For text files, append as before
            existing_content = download_hf_file(
                file_name=file_name,
                repo_id=repo_id,
                token=token,
                save_path=save_path,
            )
            _logger.info(f"Successfully downloaded existing {file_name}")

            # Format the separator with the current date if needed
            if "{date}" in separator:
                separator = separator.format(
                    date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )

            # Combine the existing and new content
            combined_content = existing_content + separator + new_content
            _logger.info(
                f"Combined content created with {len(existing_content)} + {len(new_content)} characters"
            )

    except Exception as e:
        _logger.warning(
            f"Could not download existing file: {e}. Creating new file."
        )
        combined_content = new_content

    # Update the file with the combined content
    update_hf_file(
        file_name=file_name,
        repo_id=repo_id,
        token=token,
        new_content=combined_content,
        save_path=save_path,
    )
    _logger.info(f"File {file_name} updated in repository {repo_id}")


def update_hf_file(
    file_name: str, repo_id: str, token: str, new_content, save_path: str
) -> None:
    """
    Update a file's content in a Hugging Face repository.

    Args:
        file_name: Name of the file to update
        repo_id: Hugging Face repository ID
        token: Authentication token
        new_content: Content to update (string for text, dict for JSON)
        save_path: Local path to save the file temporarily
    """
    _logger.info(f"Updating {file_name} for repo: {repo_id}")

    # Ensure the local directory exists
    save_path = Path(save_path) / file_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle content based on type
    with open(save_path, "w") as file:
        if isinstance(new_content, dict):
            import json

            json.dump(new_content, file, indent=2)
        else:
            file.write(new_content)

    _logger.info(f"{file_name} updated and saved to: {save_path}")

    # Update the file content on Hugging Face
    upload_hf_file(save_path, file_name, repo_id, token, "dataset")

    # Delete the file from the save_path
    os.remove(save_path)
    _logger.info(f"{file_name} deleted from local path: {save_path}")


def auth_check(repo_id, hf_env_var="HF_API_KEY"):
    # check repo
    client = HfApi()
    repo_id = repo_id
    token = os.environ[hf_env_var]

    try:
        client.dataset_info(repo_id, token=token)
        return True
    except Exception as e:
        _logger.warning(f"Repo {repo_id} not found. {e}")
        return False


def upload_dataset(config: dict, repo_name: str, results_path: Path) -> None:
    """
    Upload a dataset to a Hugging Face repository.

    Args:
        config (dict): The configuration dictionary for the dataset.
        repo_name (str): The name of the Hugging Face repository.
        results_path (Path): The path to the dataset results directory.

    Returns:
        None
    """
    _logger.info(f"Uploading dataset to repo: {repo_name}")

    dataset = load_from_disk(results_path)
    dataset.push_to_hub(repo_name, token=True)

    _logger.info(f"Dataset uploaded to {repo_name}")


def add_to_main_dataset(
    repo_name: str, results_path: Path, main_hf_dataset: str
) -> None:
    """
    Add a dataset to the main Hugging Face dataset repository.

    Args:
        repo_name (str): The name of the Hugging Face repository.
        results_path (Path): The path to the dataset results directory.
        main_hf_dataset (str): The main Hugging Face dataset repository name.

    Returns:
        None
    """
    _logger.info(
        f"Adding dataset from {repo_name} to main dataset: {main_hf_dataset}"
    )

    dataset = load_from_disk(results_path)

    if not auth_check(main_hf_dataset):
        create_repo(main_hf_dataset, repo_type="dataset")
        time.sleep(5)

    if main_hf_dataset == "iberbench/iberbench_all":
        # for the main repo only add the train split:
        train_split = DatasetDict({"train": dataset["train"]})
        train_split.push_to_hub(
            main_hf_dataset, config_name=repo_name.split("/")[-1]
        )
    else:
        dataset.push_to_hub(
            main_hf_dataset, config_name=repo_name.split("/")[-1]
        )

    _logger.info(f"Dataset from {repo_name} added to {main_hf_dataset}")


def extract_dataset_details(repo_path: str, subset_name: str = None) -> dict:

    dataset = load_dataset(path=repo_path, name=subset_name)

    splits = dataset.keys()  # Train, validation, test splits
    total_size = sum(len(dataset[split]) for split in splits)

    features = dataset["train"].features

    details = {
        "splits": {split: len(dataset[split]) for split in splits},
        "fields": {name: str(info) for name, info in features.items()},
        "size": total_size,
    }

    return details
