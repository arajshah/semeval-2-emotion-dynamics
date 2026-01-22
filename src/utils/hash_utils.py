from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: dict) -> str:
    encoded = stable_json_dumps(config).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
