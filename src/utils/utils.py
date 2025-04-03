import os
from pathlib import Path

import pandas as pd


def load_url_content(url: str):
    """Load text content from a URL using LangChain's UnstructuredURLLoader."""
    loader = UnstructuredURLLoader(urls=[url])
    docs = loader.load()

    if not docs:
        return "Failed to retrieve content"

    return docs[0].page_content


def find_files_with_suffix(base_path: Path, base_name: str):
    pattern = f"{base_name}-*"
    files = list(base_path.glob(pattern))

    # Get exact pattern
    exact = base_path / base_name
    if exact.exists():
        files.append(base_path / base_name)

    files = [file.name for file in files]
    return files


def group_datasets(dfs):
    grouped_datasets = {}
    for file_name, dataset in dfs.items():
        suffix = file_name.split("_")[-1].split(".")[0].lower()
        prefix = file_name.split("_")[0].lower()
        dataset.columns = ["ID"] + [prefix] + list(dataset.columns[2:])
        if suffix not in grouped_datasets:
            grouped_datasets[suffix] = []
        grouped_datasets[suffix].append(dataset)
    return grouped_datasets


def merge_datasets(config, dfs, merge_col):
    merged_datasets = []
    for suffix, ds_list in dfs.items():
        merged_ds = ds_list[0]
        for ds in ds_list[1:]:
            merged_ds = pd.merge(
                merged_ds, ds, on=merge_col, suffixes=("", "_drop")
            )
            merged_ds.drop(
                [col for col in merged_ds.columns if "drop" in col],
                axis=1,
                inplace=True,
            )

        if config["normalizer"].get("language_var", False):
            merged_ds["language_variation"] = suffix

        merged_datasets.append(merged_ds)

    return merged_datasets


def get_files_from_dir(dir_path):
    """
    Get a list of files from the specified directory.

    Args:
        dir_path (str): Path to the directory.

    Returns:
        list: List of file paths.
    """
    files = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            files.append(file_path)
    return files
