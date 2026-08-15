"""Shared data schemas used across ingestion, retrieval, and eval."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Agency = Literal["bir", "philhealth", "pagibig"]
Language = Literal["en", "tl", "taglish"]
QuestionType = Literal["factual", "multi_hop", "unanswerable", "ambiguous"]


class SourceDoc(BaseModel):
    doc_id: str
    agency: Agency
    title: str
    url: str
    local_path: str
    doc_type: Literal["pdf", "html"]
    license_note: str = "Public domain: Philippine gov't work, IP Code Sec. 176"


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    agency: Agency
    title: str
    url: str
    text: str
    chunk_size: int
    overlap: int
    position: int


class EvalItem(BaseModel):
    qid: str
    question: str
    language: Language
    question_type: QuestionType
    gold_answer: str | None = None
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_doc_ids: list[str] = Field(default_factory=list)
    agency: Agency
    notes: str = ""


class RetrievedChunk(BaseModel):
    chunk_id: str
    score: float
    rank: int


class RunResult(BaseModel):
    qid: str
    retrieved: list[RetrievedChunk]
    generated_answer: str | None = None
    latency_ms: float | None = None
