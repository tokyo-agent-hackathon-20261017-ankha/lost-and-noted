# lost-and-noted
For Tokyo Agent Hackathon 20261017. Batch classification and recording of lost items from hotel rooms.

## Run

```sh
./run.sh          # reads AIAND_API_KEY from .env, serves http://localhost:8000
```

Staff flow: photograph items → AI reads the room number from the paper in the photo
and classifies → company-property / valuable alerts → guest contact auto-looked-up
from `guests.json` → one-tap email send. Room number is editable if misread.
Statuses are tap-cycled (in storage → in safe → to police box → returned).

`guests.json` is the demo guest registry (room number → name/email) standing in for
the hotel PMS query. Email sending uses the local SMTP relay — configure
`HOTEL_SMTP_HOST`, `HOTEL_SMTP_PORT`, `HOTEL_FROM_EMAIL` in `.env`.

`references/` holds labeled company-property hint images (e.g. the room key card)
sent to the model alongside room photos — not staff-facing. Test photos live in
`assets/images/`.

## Validate

```sh
.venv/bin/python validate.py expectations.json
```

Checks stored records against `expectations.json` (must/must-not contain, item
counts, room number, schema sanity). Exit 0 = pass.

