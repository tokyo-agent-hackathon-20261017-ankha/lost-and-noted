"""Pydantic schemas for lost-item records."""
import json
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

ItemClass = Literal[
    "expirable",
    "non_expirable",
    "flammable",
    "company_property",
    "domestic_currency",
    "foreign_currency",
    "valuable",
]

# Default lifecycle; staff may define additional statuses per item (editable labels).
DEFAULT_STATUSES = ["in_storage", "in_safe", "to_policebox"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


class Item(BaseModel):
    id: str = Field(default_factory=new_id)
    created: datetime = Field(default_factory=utcnow)
    last_updated: datetime = Field(default_factory=utcnow)
    item_class: ItemClass = Field(alias="class")
    type: str  # e.g. "brush", "key", "watch" — model inference
    status: str = DEFAULT_STATUSES[0]  # free-form; staff-editable labels

    model_config = {"populate_by_name": True}


class SessionRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    created: datetime = Field(default_factory=utcnow)
    last_updated: datetime = Field(default_factory=utcnow)
    room_number: Optional[int] = None  # inferred from the room-number paper in the image
    images: list[str] = []  # urls under /images/
    items: list[Item] = []


class StatusUpdate(BaseModel):
    status: str


class RoomUpdate(BaseModel):
    room_number: int


class EmailSend(BaseModel):
    to: str
    body: str


def send_email(to: str, subject: str, body: str) -> None:
    """Send via local SMTP relay (e.g. Postfix on localhost:25)."""
    msg = EmailMessage()
    msg["From"] = os.environ.get("HOTEL_FROM_EMAIL", "lost-and-found@hotel.local")
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(
        os.environ.get("HOTEL_SMTP_HOST", "localhost"),
        int(os.environ.get("HOTEL_SMTP_PORT", "25")),
        timeout=15,
    ) as smtp:
        smtp.send_message(msg)


def lookup_guest(room_number: int) -> dict | None:
    """Query guest contact by room number from guests.json (demo registry)."""
    path = Path("guests.json")
    if not path.exists():
        return None
    guests = json.loads(path.read_text())
    return guests.get(str(room_number))
