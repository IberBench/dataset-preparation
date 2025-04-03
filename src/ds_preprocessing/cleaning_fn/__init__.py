from typing import Callable

from .classification_norm import standard_classification_normalizer
from .hf_repo_norm import hf_repo_normalizer
from .tass2020_sentiment import normalize_tass2020_sentiment
from .vaxxstance_2020 import clean_vaxxstance

# register here new cleaning functions

cleaning_registry: dict[str, Callable] = {
    "vaxxstance": clean_vaxxstance,
    "tass2020_sentiment": normalize_tass2020_sentiment,
    "classification": standard_classification_normalizer,
    "hf_repo": hf_repo_normalizer,
}
