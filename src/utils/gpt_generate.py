import requests
import tiktoken
from bs4 import BeautifulSoup
from openai import OpenAI

from src.utils.logging import get_logger
from src.utils.preprocessing import clean_url_text

_logger = get_logger(__name__)


class GPTClient:
    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
        json_mode: bool = False,
        truncate_max_tokens: int = 500,
        gpt_timeout: int = 20,
    ):
        """
        Initializes the client with the provided API key and model configuration.

        Parameters:
        - openai_api_key (str): OpenAI API key obtained from the environment variable.
        - model (str): Model identifier, default is 'gpt-4o-mini'.
        - json_mode (bool): Switch used to include the "response_format" param as a json_object. This doesn't work on GPT-4.
        - truncate_max_tokens (int): Maximum number of tokens after truncation on the examples used for in-context learning.
        - gpt_timeout (int): The timeout for the OpenAI client in seconds.
        """
        self.client = OpenAI(
            organization=None, api_key=openai_api_key, timeout=gpt_timeout
        )

        self.generation_params = {
            "model": model,
            "max_tokens": 1,  # The client automatically fills to model's max length
            "temperature": 0,
        }

        # Compatible with models allowing json mode
        if json_mode:
            self.generation_params["response_format"] = {"type": "json_object"}

        self.tokenizer = tiktoken.encoding_for_model(model)

        self.truncate_max_tokens = truncate_max_tokens

    def extract_url_content(self, url: str) -> str:
        """
        Extracts and cleans the content from a given URL.

        Args:
            url (str): The URL to extract content from.

        Returns:
            str: The cleaned content extracted from the URL.
        """
        _logger.info(f"Extracting content from URL: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            _logger.warning(f"Failed to fetch URL content: {e}")
            return ""

        soup = BeautifulSoup(response.text, "html.parser")
        page_content = soup.get_text(separator="\n", strip=True)

        clean_content = clean_url_text(page_content)

        return clean_content

    def summarize_url(self, url: str, prompt: str) -> str:
        """
        Summarizes the content of a given URL using a prompt.

        Args:
            url (str): The URL to summarize content from.
            prompt (str): The prompt to use for summarization.

        Returns:
            str: The summarized content.
        """
        _logger.info(f"Summarizing content from URL: {url}")
        page_content = self.extract_url_content(url)

        with open("prompts/url_summarizer.txt", "r") as file:
            prompt_template = file.read()

        prompt = prompt_template.format(page_content=page_content)

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}]
            )
            summary = response.choices[0].message["content"]
        except Exception as e:
            _logger.warning(f"Failed to generate summary: {e}")
            return ""

        return summary

    def generate(
        self,
        prompt_template: str,
    ) -> str:

        _logger.info(f"Generating model card...")

        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_template}],
            model=self.generation_params["model"],
        )

        return response.choices[0].message.content
