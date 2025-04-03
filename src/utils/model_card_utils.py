import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.models.model_card_config import DatasetCard


def populate_template(dataset_card: DatasetCard):
    """Fill the dataset card template with structured output."""

    template = f"""# Dataset Card for {dataset_card.dataset_id}

> This dataset card has been automatically generated with GPT-4o 🤖.
> If you find any issue, please, contact with any IberBench member to address it.

{dataset_card.dataset_summary}

## Dataset Details

### Dataset Description

{dataset_card.dataset_description}

- **Created by:** {dataset_card.creators}
- **Funded by:** {dataset_card.funded_by}
- **Shared by:** {dataset_card.shared_by}
- **Languages:** {dataset_card.languages}
- **License:** {dataset_card.license}

### Dataset Sources

- **Repository:** {dataset_card.repo}
- **Paper:** {dataset_card.paper}

## Dataset Composition

### Data Fields

{dataset_card.data_fields}

### Data Splits

{dataset_card.data_splits}

### Data Size

{dataset_card.data_size}

### Data Collection Process

{dataset_card.data_collection_process}

## Uses

### Intended Uses

{dataset_card.intended_uses}

### Out-of-Scope Uses

{dataset_card.out_of_scope_uses}

## Citation

**BibTeX:**  
{dataset_card.citation_bibtex}

**APA:**  
{dataset_card.citation_apa}

## Dataset Authors

{dataset_card.dataset_card_authors}

## Dataset Contact

{dataset_card.dataset_card_contact}
"""
    return template


def load_content_from_urls(urls: list):
    """Load and concatenate content from multiple URLs."""
    print(f"Loading content from {len(urls)} URLs...")
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()

    if not docs:
        return None

    # Concatenate all documents' content
    all_content = "\n\n".join([doc.page_content for doc in docs])
    print(
        f"Loaded {len(docs)} documents, total content length: {len(all_content)}"
    )
    return all_content


def split_content_into_chunks(content: str, chunk_size=6000, chunk_overlap=200):
    """Split content into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(content)
    print(f"Split content into {len(chunks)} chunks")
    return chunks


def extract_data_from_response(response):
    """Extract structured data from an LLM response with tool calls."""
    if not hasattr(response, "tool_calls") or not response.tool_calls:
        print("No tool calls found in response")
        return None

    tool_call = response.tool_calls[0]

    # Handle different response formats
    if isinstance(tool_call, dict):
        if "args" in tool_call:
            return tool_call["args"]
        elif "function" in tool_call and "arguments" in tool_call["function"]:
            import json

            arguments_str = tool_call["function"]["arguments"]
            return (
                json.loads(arguments_str)
                if isinstance(arguments_str, str)
                else arguments_str
            )
        else:
            print(f"Unexpected tool_call format: {tool_call.keys()}")
            return None
    else:
        # Object-style format
        return tool_call.args


def update_dataset_card(dataset_card, extracted_data, all_fields):
    """Update a dataset card with newly extracted information."""
    updates_made = 0

    for field in all_fields:
        if field in extracted_data:
            current_value = getattr(dataset_card, field)
            new_value = extracted_data[field]

            # Only update if the new value is meaningful and current value is default/empty
            is_current_default = (
                current_value == ""
                or current_value == "Dataset ID"
                or current_value == "[More Information Needed]"
            )

            is_new_meaningful = (
                new_value != ""
                and new_value != "Dataset ID"
                and new_value != "[More Information Needed]"
            )

            if is_current_default and is_new_meaningful:
                print(
                    f"Updating field '{field}' with value: {new_value[:30]}..."
                )
                setattr(dataset_card, field, new_value)
                updates_made += 1

    return updates_made


def process_chunk(chunk_text, structured_llm, dataset_card, all_fields):
    """Process a single content chunk and update the dataset card."""
    # Create a prompt for extracting fields
    prompt = ChatPromptTemplate.from_template(
        """Extract information from the provided text to fill a dataset card. 
        Be thorough and comprehensive in your extraction.
        For any field where information isn't available in the text, use "[More Information Needed]".
        
        TEXT:
        {text}
        """
    )

    try:
        # Get structured output
        chain = prompt | structured_llm
        response = chain.invoke({"text": chunk_text})

        # Extract data from response
        extracted_data = extract_data_from_response(response)
        if not extracted_data:
            return 0

        # Update the dataset card
        updates = update_dataset_card(dataset_card, extracted_data, all_fields)
        return updates

    except Exception as e:
        print(f"Error processing chunk: {e}")
        import traceback

        traceback.print_exc()
        return 0


def generate_dataset_card_from_urls(urls: list):
    """Generate a dataset card by loading and processing content from multiple URLs."""
    # Load content from all URLs
    all_content = load_content_from_urls(urls)
    if not all_content:
        return {"error": "Failed to retrieve content from any URL"}

    # Initialize an empty dataset card
    dataset_card = DatasetCard()

    # Get all field names from DatasetCard
    all_fields = [
        f
        for f in dir(dataset_card)
        if not f.startswith("_") and not callable(getattr(dataset_card, f))
    ]

    # Split content into chunks
    chunks = split_content_into_chunks(all_content)

    # Create LLM with tools
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    structured_llm = llm.bind_tools([DatasetCard])

    # Process each chunk
    total_updates = 0
    for i, chunk_text in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} for all fields")
        updates = process_chunk(
            chunk_text, structured_llm, dataset_card, all_fields
        )
        total_updates += updates
        time.sleep(1)  # Rate limiting

    print(f"Total fields updated: {total_updates}")
    return dataset_card
