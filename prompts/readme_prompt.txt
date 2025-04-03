# TASK DEFINITION:
Your task is to use the specific configs, dataset info, URL data and model card template to create a custom model card for the dataset in hand.

# RULES:
1. Remove all "[optional]" placeholders from the template.
2. Do not include the description of the sections.
3. Ensure the dataset card follows the provided template structure without unnecessary omissions or redundancies.
4. Use concise and specific language for each section, filling in all available details accurately.

# SPECIFIC CONFIG:
- **Task Name:** {task_name}
- **Workshop and Year:** {workshop} {year}

# DATASET INFO
{dataset_info}

# URL DATA:
{url_content}

# ADDITIONAL RULES:
1. If certain fields (e.g., `citation`, `license`, `languages`) are missing, note them as `"Not Provided"` instead of leaving them blank.
2. Ensure all content adheres to proper markdown syntax.
3. Always use key-value formatting when listing multiple items (e.g., "field: value").
4. Double-check that the dataset structure and feature descriptions are accurate and reflect the dataset info.

# OUTPUT FORMAT
The output should be structured as follows:
1. **First Part of the Output:** Provide a summary or preamble based on `{{output_part_one}}`.
2. **Generated Model Card:** Fully generate the model card using the template and provided information.

## FIRST PART OF THE OUTPUT:
The first part of the output is as follows:
{output_part_one}

## MODEL CARD TEMPLATE
<start_of_template>{model_card_template}<end_of_template>

# OUTPUT