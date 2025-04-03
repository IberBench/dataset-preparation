import os
from typing import Dict, List, Union

import pandas as pd

from src.utils.utils import get_files_from_dir


class FileHandler:
    """
    A class to handle file processing for various file types.

    Attributes:
        input_dir (str): The directory containing the input files.
        input_files (List[str]): A list of file paths in the input directory.

    Methods:
        process_files() -> Dict[str, Union[pd.DataFrame, List[str]]]:
            Processes the input files and returns a dictionary of file names and their processed content.
        process_tsv(file_path: str) -> pd.DataFrame:
            Processes a TSV file and returns a DataFrame.
        process_csv(file_path: str) -> pd.DataFrame:
            Processes a CSV file and returns a DataFrame.
        process_xlsx(file_path: str) -> pd.DataFrame:
            Processes an XLSX file and returns a DataFrame.
        process_txt(file_path: str) -> pd.DataFrame:
            Processes a TXT file and returns a DataFrame.
    """

    def __init__(self, input_dir: str):
        """
        Initializes the FileHandler with the input directory.

        Args:
            input_dir (str): The directory containing the input files.
        """
        self.input_dir = input_dir
        self.input_files = get_files_from_dir(self.input_dir)

    def process_files(self) -> Dict[str, Union[pd.DataFrame, List[str]]]:
        """
        Processes the input files and returns a dictionary of file names and their processed content.

        Returns:
            Dict[str, Union[pd.DataFrame, List[str]]]: A dictionary where the keys are file names and the values are the processed content.
        """
        datasets = {}
        for file in self.input_files:
            file_name = os.path.basename(file)
            if file.endswith(".tsv"):
                datasets[file_name] = self.process_tsv(file)
            elif file.endswith(".csv"):
                datasets[file_name] = self.process_csv(file)
            elif file.endswith(".xlsx"):
                datasets[file_name] = self.process_xlsx(file)
            elif file.endswith(".txt"):
                datasets[file_name] = self.process_txt(file)
            else:
                raise ValueError(f"Unsupported file type: {file}")
        return datasets

    def process_tsv(self, file_path: str) -> pd.DataFrame:
        """
        Processes a TSV file and returns a DataFrame.

        Args:
            file_path (str): The path to the TSV file.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        return pd.read_csv(file_path, sep="\t")

    def process_csv(self, file_path: str) -> pd.DataFrame:
        """
        Processes a CSV file and returns a DataFrame.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        return pd.read_csv(file_path)

    def process_xlsx(self, file_path: str) -> pd.DataFrame:
        """
        Processes an XLSX file and returns a DataFrame.

        Args:
            file_path (str): The path to the XLSX file.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        return pd.read_excel(file_path)

    def process_txt(self, file_path: str) -> pd.DataFrame:
        """
        Processes a TXT file and returns a DataFrame.

        Args:
            file_path (str): The path to the TXT file.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        try:
            return self.process_csv(file_path)
        except:
            return self.process_tsv(file_path)
