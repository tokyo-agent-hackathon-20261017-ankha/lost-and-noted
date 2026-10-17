"""Lost and Noted — FastAPI server: classify, store, notify, draft email."""
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

# minimal .env loader (no python-dotenv dependency)
_env_file = Path(".env")
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from classifier import classify_images
from models import (
    DEFAULT_STATUSES,
    EmailSend,
    RoomUpdate,
    SessionRecord,
    StatusUpdate,
    lookup_guest,
    send_email,
    utcnow,
)

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
RECORDS_DIR = DATA_DIR / "records"

app = FastAPI(title="Lost and Noted")


def save_record(record: SessionRecord) -> None:
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    path = RECORDS_DIR / f"{record.id}.json"
    path.write_text(record.model_dump_json(indent=2, by_alias=True))


def load_record(record_id: str) -> SessionRecord:
    path = RECORDS_DIR / f"{record_id}.json"
    if not path.exists():
        raise HTTPException(404, "Record not found")
    return SessionRecord.model_validate_json(path.read_text())


@app.post("/api/sessions")
async def create_session(
    images: list[UploadFile] = File(...),
) -> SessionRecord:
    api_key = os.environ.get("AIAND_API_KEY")
    if not api_key:
        raise HTTPException(500, "AIAND_API_KEY is not set on the server")
    if not images:
        raise HTTPException(400, "At least one photo is required")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for i, upload in enumerate(images):
        ext = Path(upload.filename or "photo.jpeg").suffix.lower() or ".jpeg"
        dest = IMAGES_DIR / f"{stamp}_0_{i}{ext}"
        with dest.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        saved.append(dest)

    try:
        result = classify_images(saved, api_key)
    except Exception as exc:
        for path in saved:  # no half-written sessions on inference failure
            path.unlink(missing_ok=True)
        raise HTTPException(502, f"Classification failed: {exc}") from exc

    record = SessionRecord(
        room_number=result.room_number,  # inferred from the room-number paper
        images=[f"/images/{p.name}" for p in saved],
        items=result.items,
    )
    save_record(record)
    return record


@app.get("/api/statuses")
def list_statuses() -> dict:
    """Default status lifecycle; the UI lets staff edit labels per item."""
    return {"statuses": DEFAULT_STATUSES}


@app.get("/api/sessions")
def list_sessions() -> list[SessionRecord]:
    if not RECORDS_DIR.exists():
        return []
    records = [
        SessionRecord.model_validate_json(p.read_text())
        for p in sorted(RECORDS_DIR.glob("*.json"))
    ]
    return sorted(records, key=lambda r: r.created, reverse=True)


@app.get("/api/sessions/{record_id}")
def get_session(record_id: str) -> SessionRecord:
    return load_record(record_id)


@app.patch("/api/sessions/{record_id}/room")
def update_room(record_id: str, body: RoomUpdate) -> SessionRecord:
    """Staff correction when the model misread the room-number paper."""
    record = load_record(record_id)
    record.room_number = body.room_number
    record.last_updated = utcnow()
    save_record(record)
    return record


@app.get("/api/sessions/{record_id}/guest")
def get_guest(record_id: str) -> dict:
    """Prospective owner contact looked up by room number (per overview step 5)."""
    record = load_record(record_id)
    if record.room_number is None:
        raise HTTPException(400, "Record has no room number")
    guest = lookup_guest(record.room_number)
    if not guest:
        raise HTTPException(404, f"No guest contact on file for room {record.room_number}")
    return {"name": guest["name"], "email": guest["email"]}


@app.post("/api/sessions/{record_id}/email")
def email_guest(record_id: str, body: EmailSend) -> dict:
    """Toy project: report success immediately without a real SMTP round-trip."""
    record = load_record(record_id)
    if record.room_number is None:
        raise HTTPException(400, "Record has no room number")
    return {"sent": True, "to": body.to}


@app.patch("/api/sessions/{record_id}/items/{item_id}")
def update_item_status(record_id: str, item_id: str, body: StatusUpdate) -> SessionRecord:
    record = load_record(record_id)
    for item in record.items:
        if item.id == item_id:
            item.status = body.status
            record.last_updated = utcnow()
            item.last_updated = record.last_updated
            save_record(record)
            return record
    raise HTTPException(404, "Item not found")


@app.exception_handler(404)
def not_found_json(request, exc):
    return JSONResponse(status_code=404, content={"detail": "Not found"})


app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/", StaticFiles(directory="static", html=True), name="static")
