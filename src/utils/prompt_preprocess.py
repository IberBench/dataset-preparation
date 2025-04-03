import os
from pathlib import Path

from src.utils import (
    GPTClient,
    config_parser,
    dataset_results_path,
    download_hf_file,
    extract_dataset_details,
)


class PromptPreparation:
    """
    A class to prepare the necessary data for generating a prompt using GPT.

    Attributes:
        config (dict): Configuration details for the dataset.
        gpt_client (GPTClient): An instance of the GPTClient class.
        results_path (Path): Path to the results directory.

    Methods:
        prepare() -> str:
            Prepares the data and returns the formatted prompt string.
        readme() -> str:
            Downloads the README content from the Hugging Face repository.
        dataset_details() -> dict:
            Extracts dataset details from the Hugging Face repository.
        url_content() -> str:
            Extracts and cleans the content from a given URL.
    """

    def __init__(self, config: dict, gpt_client: GPTClient):
        """
        Initializes the PromptPreparation class with the provided configuration and GPT client.

        Args:
            config (dict): Configuration details for the dataset.
            gpt_client (GPTClient): An instance of the GPTClient class.
        """
        self.config = config
        self.gpt_client = gpt_client
        self.results_path = dataset_results_path(
            Path("results"),
            task_name=config["dataset"]["task_name"],
            workshop=config["dataset"]["workshop"],
            year=config["dataset"]["year"],
            language=config["normalizer"]["language"],
        )

        self.aggregated_repo = config["huggingface_config"].get(
            "aggregated_subset", ""
        )
        if self.aggregated_repo:
            self.repo_id = self.aggregated_repo
            self.subset_name = config["huggingface_config"]["main_subset"]

        else:
            self.repo_id = self.config["huggingface_config"]["repo_path"]
            self.subset_name = None

    def prepare(self) -> str:
        readme_content = self.readme()
        dataset_details = self.dataset_details()
        url_content = self.url_content()

        return config_parser(
            config_details=self.config,
            dataset_details=dataset_details,
            readme_content=readme_content,
            url_content=url_content,
        )

    def readme(self) -> str:

        return download_hf_file(
            file_name="README.md",
            repo_id=self.repo_id,
            token=os.environ["OPENAI_SYMANTO_TOKEN"],
            save_path=self.results_path,
        )

    def dataset_details(self) -> dict:
        return extract_dataset_details(
            repo_path=self.repo_id,
            subset_name=self.subset_name,
        )

    def url_content(self) -> str:
        return self.gpt_client.extract_url_content(
            self.config["huggingface_config"]["task_url"]
        )
