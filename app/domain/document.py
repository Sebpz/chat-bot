from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field


class Paragraph(BaseModel):
    kind: Literal["paragraph"] = "paragraph"
    text: str
    page_number: int | None = None
    heading: str | None = None


class Table(BaseModel):
    kind: Literal["table"] = "table"
    text: str
    page_number: int | None = None
    heading: str | None = None


class Figure(BaseModel):
    kind: Literal["figure"] = "figure"
    caption: str | None = None
    page_number: int | None = None
    heading: str | None = None


Element = Union[Paragraph, Table, Figure]


class Section(BaseModel):
    heading: str
    level: int = 0
    elements: list[Element] = Field(default_factory=list)


class Document(BaseModel):
    document_id: str
    filename: str
    sections: list[Section] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
