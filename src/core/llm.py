"""LLM manager: extracts model archive, loads Phi-3.5-mini in 4-bit, and
exposes a LangChain-compatible HuggingFacePipeline."""

import logging
import tarfile

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline as hf_pipeline,
)

from src.core.config import settings

logger = logging.getLogger(__name__)


class LLMManager:
    """Handles model extraction, loading, and pipeline creation."""

    def __init__(self) -> None:
        self._pipeline: HuggingFacePipeline | None = None

    @property
    def pipeline(self) -> HuggingFacePipeline:
        if self._pipeline is None:
            raise RuntimeError("LLM pipeline not initialised. Call load() first.")
        return self._pipeline

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def ensure_model_extracted(self) -> None:
        """Extract the .tar.gz archive if the model directory does not exist."""
        model_dir = settings.model_dir
        archive = settings.model_archive

        if model_dir.exists() and any(model_dir.iterdir()):
            logger.info("Model already extracted at %s", model_dir)
            return

        if not archive.exists():
            raise FileNotFoundError(f"Model archive not found: {archive}")

        logger.info("Extracting model archive %s -> %s ...", archive, model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=model_dir, filter="data")

        logger.info("Model extraction complete")

    def load(self) -> None:
        """Load the quantised model and build the LangChain pipeline."""
        self.ensure_model_extracted()

        model_path = str(settings.model_dir)
        logger.info("Loading tokenizer from %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        logger.info("Loading model in 4-bit quantisation ...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=False,
        )

        pipe = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=settings.llm_max_new_tokens,
            temperature=settings.llm_temperature,
            do_sample=True,
            return_full_text=False,
        )

        self._pipeline = HuggingFacePipeline(pipeline=pipe)
        logger.info("LLM pipeline ready (4-bit, device_map=auto)")
