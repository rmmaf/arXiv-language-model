"""LLM manager: extracts model archive, loads Phi-3.5-mini in 4-bit, and
exposes both a LangChain-compatible pipeline and a cancellable generate
method that honours a ``threading.Event`` to interrupt GPU inference."""

import logging
import tarfile
import threading

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline as hf_pipeline,
)

from src.core.config import settings

logger = logging.getLogger(__name__)


class CancellableStoppingCriteria(StoppingCriteria):
    """Stops ``model.generate()`` when a ``threading.Event`` is set.

    Checked once per generated token, so cancellation latency is at most
    one forward pass.
    """

    def __init__(self, cancel_event: threading.Event) -> None:
        super().__init__()
        self._cancel_event = cancel_event

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: object,
    ) -> bool:
        return self._cancel_event.is_set()


class LLMManager:
    """Handles model extraction, loading, and pipeline creation."""

    def __init__(self) -> None:
        self._pipeline: HuggingFacePipeline | None = None
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    @property
    def pipeline(self) -> HuggingFacePipeline:
        if self._pipeline is None:
            raise RuntimeError(
                "LLM pipeline not initialised. Call load() first."
            )
        return self._pipeline

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def ensure_model_extracted(self) -> None:
        """Extract the .tar.gz if the model dir does not exist."""
        model_dir = settings.model_dir
        archive = settings.model_archive

        if model_dir.exists() and any(model_dir.iterdir()):
            logger.info("Model already extracted at %s", model_dir)
            return

        if not archive.exists():
            raise FileNotFoundError(
                f"Model archive not found: {archive}"
            )

        logger.info(
            "Extracting model archive %s -> %s ...",
            archive, model_dir,
        )
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

        self._model = model
        self._tokenizer = tokenizer

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

    def generate(
        self,
        prompt: str,
        cancel_event: threading.Event,
    ) -> str:
        """Run text generation with a cancellable stopping criteria.

        This method is **synchronous** and meant to be called inside
        ``loop.run_in_executor`` so the event-loop stays free while the
        GPU works.  The *cancel_event* is checked after every generated
        token; when set, generation stops immediately.

        Returns the generated text (excluding the prompt).
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "LLM not initialised. Call load() first."
            )

        inputs = self._tokenizer(
            prompt, return_tensors="pt",
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        stopping = StoppingCriteriaList([
            CancellableStoppingCriteria(cancel_event),
        ])

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=settings.llm_max_new_tokens,
                temperature=settings.llm_temperature,
                do_sample=True,
                stopping_criteria=stopping,
            )

        new_ids = output_ids[0][input_len:]
        text = self._tokenizer.decode(
            new_ids, skip_special_tokens=True,
        )

        if cancel_event.is_set():
            logger.info("Generation interrupted by cancel event")

        return text
