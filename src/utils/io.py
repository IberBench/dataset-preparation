import json
from pathlib import Path
from typing import Optional

from datasets import Dataset
from src.models.config import Config, TaskConfig


def read_json(path: str | Path) -> dict:
    """
    Reads a json file.

    Args:
        path (str | Path): path to the json file

    Returns:
        dict: loaded json
    """
    with open(path, "r") as fr:
        return json.load(fr)


def save_json(path: str | Path, json_dict: dict) -> None:
    """
    Saves a dict into a json file.

    Args:
        path (str | Path): path to the json file.
        json_dict (dict): dictionary to be saved.
    """
    with open(path, "w") as fw:
        json.dump(json_dict, fw, indent=4)


def load_configs(path: str | Path) -> Config:
    """
    Load config from a JSON file and validate it using Pydantic.

    Args:
        path (str | Path): Path to the config file (JSON).

    Returns:
        Config: The validated configuration object.
    """
    path = Path(path)
    if path.suffix == ".json":
        config_dict = read_json(path)
        return Config(**config_dict)
    else:
        raise ValueError("Only .json files are allowed")


def create_dataset_name(
    task_config: TaskConfig, language_variety: Optional[str] = None
) -> Path:
    """
    Creates the dataset name from the task config. Separates fields by "-", ensuring
    no field contain "-" as separator.
    """
    workshop = task_config.workshop.replace("-", "_")
    shared_task = task_config.shared_task.replace("-", "_")
    year = str(task_config.year).replace("-", "_")
    language = task_config.language.replace("-", "_")
    task_type = task_config.task_type.replace("-", "_")
    language_variety = (
        language_variety.replace("-", "_") if language_variety else None
    )

    if language_variety:
        return f"{workshop}-{shared_task}-{task_type}-{year}-{language}-{language_variety}"
    return f"{workshop}-{shared_task}-{task_type}-{year}-{task_config.language}"


def create_repo_name(prefix: str, dataset_name: str):
    return f"{prefix}/{dataset_name}"


def get_language_variety(dataset: Dataset) -> Optional[str]:
    if "language_variation" in dataset["train"].features:
        return dataset["train"]["language_variation"][0]
    return None


def dataset_results_path(
    base_path: str | Path,
    task_config: TaskConfig,
    dataset: Dataset = None,
) -> Path:
    """
    Appends to a base path a random file name.

    Args:
        base_path (str | Path): the base path of the file

    Returns:
        Path: path to the file
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    language_variety = None
    if dataset:
        language_variety = get_language_variety(dataset)

    dataset_name = create_dataset_name(task_config, language_variety)
    return base_path / dataset_name

    
def create_dataset_metadata(config: Config, dataset: Dataset):
    task_config = config.task
    task_info = task_config.model_dump()
    # Add language variety
    language_variety = get_language_variety(dataset)
    task_info["language_variety"] = language_variety

    # If this info is required it should be moved into the task_config
    # of the YAML configs. There is a lot of diversity to cover it
    # just with code. But at least, this serves as preliminary,
    # effortless apprach.
    # Add problem type
    problem_type = "classification"
    if "chunking" in task_config.task_type:
        problem_type = "chunking"
    elif "tagging" in task_config.task_type:
        problem_type = "tagging"
    elif "generation" in task_config.task_type:
        problem_type = "generation"
    elif "question_answering" in task_config.task_type:
        problem_type = "question_answering"
    elif "entailment" in task_config.task_type:
        problem_type = "textual_entailment"
    elif "reading_comprehension" in task_config.task_type:
        problem_type = "reading_comprehension"
    elif "summarization" in task_config.task_type:
        problem_type = "summarization"
    task_info["problem_type"] = problem_type

    # Add num labels and labels if classification
    if problem_type == "classification":
        label_set = list(set(dataset["test"]["label"]))
        task_info["num_labels"] = len(label_set)
        label_mapping = config.mapping.get("label")
        if label_mapping is None and len(label_set) == 2:
            label_set = ["no", "yes"]
        else:
            if label_mapping:
                label_mapping = {v: k for k, v in label_mapping.items()}
                label_set = [label_mapping[label] for label in label_set]

        task_info["labels"] = label_set
            
    return task_info
