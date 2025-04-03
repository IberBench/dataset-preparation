import json

from ftfy import fix_text


def clean_labels(labels: str) -> str:
    if type(labels) != str:
        labels = str(labels)
    # remove blank spaces
    labels = labels.strip()
    # set to lowercase
    labels = labels.lower()

    return labels


def clean_text(text: str) -> str:
    """
    Cleans a text from datasets

    Args:
        text (str): a text

    Returns:
        str: a cleaned text
    """
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    text = text.replace(":", " ")
    text = " ".join(text.split())
    return text


def clean_url_text(text: str) -> str:
    """
    Cleans text extracted from a URL (e.g., HTML pages).

    Args:
        text (str): Raw text extracted from a URL.

    Returns:
        str: A cleaned text.
    """
    import re

    # Remove excessive newlines, tabs, and carriage returns
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # Replace non-breaking spaces and other Unicode whitespace
    text = text.replace("\xa0", " ").replace("\u200b", " ")

    # Remove excessive spaces and normalize
    text = " ".join(text.split())

    # Remove URLs if present within the text
    text = re.sub(r"http[s]?://\S+", "", text)

    # Remove unnecessary colons (if applicable)
    text = text.replace(":", " ")

    # Optional: Remove HTML entities (e.g., &amp; -> &, &lt; -> <)
    text = re.sub(r"&[a-z]+;", " ", text)

    # Optional: Remove special characters or HTML artifacts
    text = re.sub(r"[<>]", "", text)

    return text


def fix_encoding(text: str) -> str:
    """
    Fixes encoding issues in a text

    Args:
        text (str): a text

    Returns:
        str: a text with fixed encoding
    """
    try:
        text = fix_text(text)
    except Exception:
        text = "no available text"
    # Additional cleaning steps can be added here if needed
    return text


def config_parser(
    config_details: dict,
    dataset_details: dict,
    readme_content: str,
    url_content: str,
) -> str:
    """
    Parses a configuration dictionary and fills in the placeholders in the template.

    Args:
        config_details (dict): A dictionary with configuration details.
        template_path (str): Path to a template file with placeholders.

    Returns:
        str: A string with the filled placeholders.
    """
    with open("prompts/readme_prompt.txt", "r") as file:
        prompt_template = file.read()

    with open("prompts/model_card_template.txt", "r") as file:
        model_card_template = file.read()

    return prompt_template.format(
        output_part_one=readme_content,
        model_card_template=model_card_template,
        url_content=url_content,
        dataset_info=json.dumps(dataset_details, indent=2),
        **config_details["dataset"],
    )
