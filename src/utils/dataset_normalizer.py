from src.utils.preprocessing import clean_labels, fix_encoding
from src.utils.utils import get_files_from_dir


class DatasetNormalizer:
    def __init__(self, config=dict):
        self.train_files = config["dataset"]["train_files"]
        self.test_files = config["dataset"]["test_files"]
        self.language_var = config["normalizer"].get("language_var", False)
        self.language = config["task"]["language"]
        self.input_cols = config["normalizer"]["input_cols"]
        self.output_col = config["normalizer"]["output_col"]
        self.mapping = config["mapping"]
        self.keep_columns = config["normalizer"]["keep_columns"]

    def normalize_texts(self, df):
        for text_column in self.input_cols:
            df[text_column.lower()] = df[text_column.lower()].apply(
                fix_encoding
            )
        return df

    def normalize_column(self, df):
        for column_name, mapping in self.mapping.items():
            if column_name == "desired_column_mapping":
                df.rename(columns=mapping, inplace=True)
            else:
                # Determine the type of keys in the mapping
                if all(isinstance(k, int) for k in mapping.keys()):
                    df[column_name] = df[column_name].astype(int)
                elif all(isinstance(k, str) for k in mapping.keys()):
                    df[column_name] = df[column_name].astype(str)

                df[column_name] = (
                    df[column_name].map(mapping).fillna(df[column_name])
                )
                df[column_name] = df[column_name].str.lower()
        return df

    def add_language_column(self, df):
        if "language" not in df.columns:
            df["language"] = self.language
        return df

    def add_language_variation_column(self, dfs, dir_path):
        files = get_files_from_dir(dir_path)
        for file, ds in zip(files, dfs):
            suffix = file.split("_")[-1].split(".")[0].lower()
            ds["language_variation"] = suffix
        return dfs

    def standard_cleanup(self, df):
        # clean cols just in case
        df.columns = [col.lower().replace(" ", "") for col in df.columns]
        # add the language column
        df = self.add_language_column(df)
        # decode texts
        df = self.normalize_texts(df)
        # clean the labels col
        df[self.output_col] = df[self.output_col].apply(clean_labels)
        # check for possible mappings
        if self.mapping:
            df = self.normalize_column(df)
        # check for correct language
        if self.language_var:
            unique_variations = df["language_variation"].unique()
            if any(
                self.language in variation for variation in unique_variations
            ):
                df = df[df["language_variation"].str.contains(self.language)]
        # keep only the columns we want
        df = df[self.keep_columns]

        return df
