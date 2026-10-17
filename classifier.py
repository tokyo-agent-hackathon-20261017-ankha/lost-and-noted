"""ai& vision inference: classify lost items from room photos."""
import base64
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from models import Item, new_id, utcnow

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

Be exhaustive: one Item object per detected item IN THE ROOM IMAGES only — \
never output the reference images as items. If the same item type appears \
multiple times, list each occurrence separately. The room number is written on \
a paper in the image: read it carefully and report it as room_number (digits \
only; null only if genuinely unreadable).\
"""


class Classification(BaseModel):
    items: list[Item]
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


def classify_images(image_paths: list[Path], api_key: str) -> Classification:
    """Upload images and classify their contents via ai& vision."""
    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    reference_parts, reference_ids = _load_reference_parts(client)

    content: list[dict] = list(reference_parts)
    content.append({"type": "text", "text": "Classify all lost items in these images."})
    file_ids: list[str] = []
    for path in image_paths:
        uploaded = client.files.create(
            file=open(path, "rb"), purpose="vision"
        )
        file_ids.append(uploaded.id)
        content.append({"type": "file", "file": {"file_id": uploaded.id}})

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

    result = response.choices[0].message.parsed
    if not result:
        raise ValueError("Model returned no parsed result")

    # Model may fill item ids/dates oddly; normalize server-side.
    now = utcnow()
    for item in result.items:
        item.id = new_id()
        item.created = item.last_updated = now
    return result
