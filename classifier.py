"""ai& vision inference: classify lost items from room photos."""
import base64
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal

from models import DEFAULT_STATUSES, Item, new_id, utcnow

# Statuses the model may assign at inference time (staff can edit labels later).
InferenceStatus = Literal["in_storage", "in_safe", "to_policebox"]

MODEL = "zai-org/glm-5.3-flash"  # cheapest vision-capable model on ai&
BASE_URL = "https://api.aiand.com/v1"

SYSTEM_PROMPT = """\
You are a hotel lost-and-found classifier. The LAST image(s) show items found \
in a checked-out hotel room (possibly with a paper showing the room number). \
Any images labeled "REFERENCE" that come before them are examples of the \
hotel's own property — use them only as a visual hint for identification.

Classify EVERY distinct item you can detect in the room images into exactly one class:
- expirable: perishables, food, medicine, cosmetics (hotel keeps 1 month)
- non_expirable: ordinary belongings, clothes, books, chargers (hotel keeps 2 days \
before disposal unless valuable)
- flammable: lighters, batteries, aerosols (batch-processed, fire hazard)
- company_property: hotel property such as room key cards, towels, robes, \
umbrellas from the hotel set, remote controls (return to storage). If an object \
visibly matches a REFERENCE image, be confident and label it company_property.
- domestic_currency: Japanese yen banknotes/coins
- foreign_currency: non-JPY banknotes/coins
- valuable: high-value items such as watches, glasses/sunglasses, jewelry, \
necklaces, wallets with cards, passports, electronics (hotel keeps 1 month, \
guest is contacted)

For "type" give a short lowercase label of the object (e.g. "toothbrush", \
"key card", "lighter", "watch", "10,000 yen note").

For "status" use EXACTLY one of: "in_storage", "in_safe", "to_policebox" \
(newly found items are always "in_storage").

Be exhaustive: one Item object per detected item IN THE ROOM IMAGES only — \
never output the reference images as items. If the same item type appears \
multiple times, list each occurrence separately. The room number is written on \
a paper in the image: read it carefully and report it as room_number (digits \
only; null only if genuinely unreadable).\
"""


class InferenceItem(BaseModel):
    """Item as the model may emit it — status constrained to the default lifecycle."""
    id: str = Field(default_factory=new_id)
    created: datetime | None = None
    last_updated: datetime | None = None
    item_class: Literal[
        "expirable", "non_expirable", "flammable", "company_property",
        "domestic_currency", "foreign_currency", "valuable",
    ] = Field(alias="class")
    type: str
    status: InferenceStatus = "in_storage"

    model_config = {"populate_by_name": True}


class Classification(BaseModel):
    items: list[InferenceItem]
    room_number: int | None = Field(default=None, alias="room_number")

    model_config = {"populate_by_name": True}


def _image_part(path: Path) -> dict:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _load_reference_parts(client: OpenAI) -> tuple[list[dict], list[str]]:
    """Load labeled company-property reference images from references/ as hints."""
    parts: list[dict] = []
    file_ids: list[str] = []
    ref_dir = Path("references")
    if not ref_dir.is_dir():
        return parts, file_ids
    for path in sorted(ref_dir.iterdir()):
        if path.suffix.lower() not in (".jpeg", ".jpg", ".png", ".webp"):
            continue
        label = path.stem.replace("-", " ").replace("_", " ")
        parts.append({"type": "text",
                      "text": f"REFERENCE — {label} (hotel company property):"})
        uploaded = client.files.create(file=open(path, "rb"), purpose="vision")
        file_ids.append(uploaded.id)
        parts.append({"type": "file", "file": {"file_id": uploaded.id}})
    return parts, file_ids


def classify_images(image_paths: list[Path], api_key: str, log_fn=None) -> Classification:
    """Upload images and classify their contents via ai& vision."""
    log = log_fn or (lambda msg: None)
    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    log("loading reference images…")
    reference_parts, reference_ids = _load_reference_parts(client)
    log(f"references loaded: {len(reference_ids)}")

    content: list[dict] = list(reference_parts)
    content.append({"type": "text", "text": "Classify all lost items in these images."})
    file_ids: list[str] = []
    for path in image_paths:
        log(f"uploading {path.name} to ai&…")
        uploaded = client.files.create(
            file=open(path, "rb"), purpose="vision"
        )
        file_ids.append(uploaded.id)
        content.append({"type": "file", "file": {"file_id": uploaded.id}})
        log(f"file_id={uploaded.id}")

    log(f"POST /chat/completions (model={MODEL}, response_format=json_schema)…")
    t0 = time.monotonic()
    try:
        response = client.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format=Classification,
        )
    finally:
        for fid in file_ids + reference_ids:  # don't let files pile up for 30 days
            try:
                client.files.delete(fid)
            except Exception:
                pass

    usage = response.usage
    secs = time.monotonic() - t0
    log(f"inference done in {secs:.1f}s "
        f"(prompt={usage.prompt_tokens} tok, completion={usage.completion_tokens} tok)")
    cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.50) / 1_000_000
    log(f"estimated cost: ${cost:.4f}")

    result = response.choices[0].message.parsed
    if not result:
        raise ValueError("Model returned no parsed result")

    # Normalize into stored Item objects: server owns ids/timestamps, and the
    # status enum is enforced (schema-constrained, plus a hard guard here).
    now = utcnow()
    items = [
        Item(
            id=new_id(),
            created=now,
            last_updated=now,
            **{"class": item.item_class},
            type=item.type,
            status=item.status if item.status in DEFAULT_STATUSES else DEFAULT_STATUSES[0],
        )
        for item in result.items
    ]
    result.items = items
    return result
