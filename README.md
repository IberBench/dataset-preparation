<h3 align="center">
    <p><b> IBERBENCH </b></p>
</h3>

| metadata  |                                                                                                                             |
| --------- | --------------------------------------------------------------------------------------------------------------------------- |
| Authors   | Ian Borrego, Álvaro Romo, Jose Gonzalez                                                                                                   |

# 📖 Introduction

🌀 **ivace_datasets** is intended to process datasets through a multi-step pipeline, including normalization, individual uploads, evaluation, model card creation, and final aggregation. The current pipeline consists of the following parts:

1. **Dataset normalization**
2. **Upload dataset to HuggingFace**
3. **Include LmevalHarness YAML and test**
4. **Include model card with agent**

## 👻 Usage with all args

```bash
Usage: python -m ivace_datasets.cli [ARGS]...

# Normalize and save a dataset based on the provided configuration:
$ python -m src.cli normalizer-and-save --config-path configs/to_hf/test/vaxxstance2021.json --results-path results

# Upload datasets to Hugging Face individually:
$ python -m src.cli upload-ds-to-huggingface --config-path configs/to_hf/test/ --main-hf-dataset iberbench/dataset-draft --dataset-path results --add-to-main-ds True

# Create a model card using GPT:
$ python -m src.cli create-model-card --config-path configs/to_hf/test/ --gpt-model gpt-4o-mini

# Upload datasets to Hugging Face with aggregation:
$ python -m src.cli upload-to-hf --config-path configs/to_hf/test/ --dataset-path datasets/tass_2020/emotion_detection


ARGS:
normalizer_and_save
    --config-path: Path to the configuration file. [default: configs/to_hf/test/vaxxstance2021.json]
    --results-path: Path to save the cleaned dataset. [default: results]
upload_ds_to_huggingface
    --config-path: Path to the configuration directory. [default: configs/to_hf/test/]
    --main-hf-dataset: Main Hugging Face dataset to aggregate results. [default: iberbench/dataset_draft]
    --dataset-path: Path to the results directory. [default: results]
    --add-to-main-ds Bool which enables adding the dataset to an aggregation already in HF [default: True]
create_model_card
    --config-path: Path to the configuration directory. [default: configs/to_hf/test/]
    --gpt-model: GPT model to use for generating the model card. [default: gpt-4o-mini]
upload_to_hf
    --config-path: Path to the configuration directory. [default: configs/to_hf/test/]
    --dataset-path: Path to the directory with the extra files you want to upload. [default: datasets/tass_2020/emotion_detection]
```

# 🚀 Pipeline Steps

## PART 1: Dataset normalization
The `normalizer_and_save` function normalizes and saves a dataset based on the provided configuration. It performs the following steps:

1. **📂 Load Configuration**:
   - Loads the configuration from the specified path.

2. **🧹 Clean Dataset**:
   - Cleans the dataset based on the configuration.
   - This configuration includes a `normalizer_fn` which parses the dataset through the function that best fits it.

3. **💾 Save Cleaned Dataset**:
   - Saves the cleaned dataset to the specified results path.

### PART 2: Upload Dataset Individually

The `upload_ds_to_huggingface` function uploads datasets to Hugging Face individually. It performs the following steps:

1. **📂 Load Configuration**:
   - Loads the configuration from the specified path.

2. **🔍 Check Repository**:
   - Checks if the repository exists on Hugging Face. If not, it creates a new repository.

3. **📤 Upload Files**:
   - Iterates through the files and directories in the specified dataset path and uploads them to the specified Hugging Face repository.

4. **📊 Upload to Aggregation (Optional)**:
   - If specified, uploads the dataset to a main aggregation repository on Hugging Face.

## PART 3: Include LmevalHarness YAML and Test

This step involves including the LmevalHarness YAML configuration and running tests to ensure the dataset meets the required standards.

📄 **Creating the YAML File**:
To create the YAML file and make it run, please follow this guide: [New Task Guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md)

🔍 **YAML Fields Details**:
For details on the specific fields of the YAML, please check this guide: [Task Guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)

🌿 **Branch for Testing**:
The branch to work on to test and later call these tasks is the following: `XXX`

In that branch, feel free to create the relevant tasks and test them with your desired models.

🛠️ **Task Formatting and Metrics**:
For different tasks, formatting, or metric elaboration, we recommend checking the tasks that have already been performed to create new YAMLs for specific tasks.

✅ **Testing and Uploading**:
Once the YAML has been forged, test it with a base model. If it runs correctly, upload it to the corresponding repo with its results using the `upload_to_hf` function.

## PART 4: Include model card with agent

The `create_model_card` function generates a model card using GPT and updates the README in the Hugging Face repository. It performs the following steps:

**🔄 Initialization**:
   - Initializes the `GPTClient` with the provided API key, access token, username, and model.

**📂 Processing Configuration Files**:
   - Iterates over each configuration file in the specified configuration directory.

**🛠️ Prepare Configuration Data**:
   - Uses the `PromptPreparation` class to prepare the necessary data for generating the prompt. This includes:
     - 📥 Downloading the README content from the specified Hugging Face repository.
     - 📊 Extracting dataset details from the Hugging Face repository.
     - 🌐 Extracting and cleaning the content from a given URL.

**📝 Generate Prompt Template**:
   - Parses the configuration and README content to create a prompt template using the `config_parser` function.

**🧠 Generate Model Card**:
   - Generates the model card using the specified GPT model.

**📄 Update README**:
   - Updates the README with the generated model card using the `update_readme` function.

# 🛠️ Development
To contribute to this project, follow these steps:

Clone the repository:
```https://github.com/IberBench/dataset-preparation.git```

Install the required dependencies:
```pip install -r requirements.txt```

Run the tests:
```python -m unittest discover -s tests```
