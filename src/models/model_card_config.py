from pydantic import BaseModel, Field


class DatasetCard(BaseModel):
    dataset_id: str = Field(
        default="Dataset ID", description="The ID or name of the dataset."
    )
    dataset_summary: str = Field(
        default="", description="A short summary of the dataset."
    )
    dataset_description: str = Field(
        default="", description="A detailed description of the dataset."
    )
    creators: str = Field(
        default="[More Information Needed]",
        description="Names of dataset creators.",
    )
    funded_by: str = Field(
        default="[More Information Needed]",
        description="Funding sources for the dataset.",
    )
    shared_by: str = Field(
        default="[More Information Needed]",
        description="Entities or individuals sharing the dataset.",
    )
    languages: str = Field(
        default="[More Information Needed]",
        description="Languages in the dataset.",
    )
    license: str = Field(
        default="[More Information Needed]", description="License information."
    )
    repo: str = Field(
        default="[More Information Needed]",
        description="Link to the dataset repository.",
    )
    paper: str = Field(
        default="[More Information Needed]",
        description="Link to any research paper about the dataset.",
    )
    data_fields: str = Field(
        default="[More Information Needed]",
        description="Description of dataset fields.",
    )
    data_splits: str = Field(
        default="[More Information Needed]",
        description="Dataset splits (train/test/validation).",
    )
    data_size: str = Field(
        default="[More Information Needed]",
        description="Size details of the dataset.",
    )
    data_collection_process: str = Field(
        default="[More Information Needed]",
        description="Methodology for data collection.",
    )
    intended_uses: str = Field(
        default="[More Information Needed]",
        description="Intended applications of the dataset.",
    )
    out_of_scope_uses: str = Field(
        default="[More Information Needed]",
        description="Misuse or unintended applications.",
    )
    citation_bibtex: str = Field(
        default="[More Information Needed]",
        description="Citation in BibTeX format.",
    )
    citation_apa: str = Field(
        default="[More Information Needed]",
        description="Citation in APA format.",
    )
    dataset_card_authors: str = Field(
        default="[More Information Needed]",
        description="Authors of the dataset card.",
    )
    dataset_card_contact: str = Field(
        default="[More Information Needed]",
        description="Contact information for inquiries.",
    )
