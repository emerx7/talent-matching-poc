"""Load config.toml into typed settings."""
from __future__ import annotations

import tomllib
from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent


class General(BaseModel):
    seed: int
    reference_date: date
    data_dir: str
    outputs_dir: str


class Generation(BaseModel):
    n_officers: int
    n_ground_truth_docs: int


class Scoring(BaseModel):
    recency_decay: float
    confidence_base: float
    confidence_step: float
    ready_now_min_fit: float
    max_gaps_for_developing: int
    max_closable_level_gap: int
    thin_bench_max_ready: int


class Extraction(BaseModel):
    mode: str
    llm_model: str
    llm_effort: str
    llm_max_tokens: int
    embedding_model: str
    suggestion_threshold: float


class Recommend(BaseModel):
    top_k: int
    tag_weight: float
    embedding_weight: float


class Fairness(BaseModel):
    attribute_seed: int
    air_threshold: float
    counterfactual_sample: int
    counterfactual_sample_llm: int


class Settings(BaseModel):
    general: General
    generation: Generation
    scoring: Scoring
    extraction: Extraction
    recommend: Recommend
    fairness: Fairness

    @property
    def data_dir(self) -> Path:
        return ROOT / self.general.data_dir

    @property
    def outputs_dir(self) -> Path:
        return ROOT / self.general.outputs_dir

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def audit_dir(self) -> Path:
        return self.data_dir / "_audit"


@lru_cache
def load_settings(path: Path | None = None) -> Settings:
    path = path or ROOT / "config.toml"
    with open(path, "rb") as f:
        return Settings.model_validate(tomllib.load(f))
