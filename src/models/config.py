from typing import Dict, List

from pydantic import BaseModel


class TaskConfig(BaseModel):
    workshop: str
    shared_task: str
    year: int
    task_type: str
    language: str
    url: List[str]


class DatasetConfig(BaseModel):
    train_files: str
    test_files: str
    hf_repo_id: str
    hf_subset: str


class NormalizerConfig(BaseModel):
    normalizer_fn: str
    language_var: bool
    input_cols: List[str]
    output_col: str
    keep_columns: List[str]


class Config(BaseModel):
    task: TaskConfig
    dataset: DatasetConfig
    normalizer: NormalizerConfig
    mapping: Dict[str, Dict[str, str]]
