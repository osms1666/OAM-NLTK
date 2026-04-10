"""Dictionary / lexicon management.

A "dictionary" is a simple mapping: category name -> list of keywords.
Bundled dictionaries ship as JSON files alongside this module.  Users can
create and save their own via the dashboard or by dropping JSON into the
``user/`` subfolder.

Quick start
-----------
>>> from oam_nltk.dictionaries import load_dictionary, list_dictionaries
>>> list_dictionaries()
['conflict', 'economy', 'environmental', 'gender', 'governance', 'health', 'oam']
>>> d = load_dictionary("oam")
>>> d.categories.keys()
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).parent
_USER = _HERE / "user"
_USER.mkdir(exist_ok=True)


@dataclass
class Dictionary:
    name: str
    categories: dict[str, list[str]] = field(default_factory=dict)
    description: str = ""

    @property
    def all_terms(self) -> list[str]:
        """Flat, deduplicated, lowercased list of every keyword."""
        return sorted({t.lower() for terms in self.categories.values() for t in terms})

    def to_json(self) -> str:
        return json.dumps(
            {"name": self.name, "description": self.description,
             "categories": self.categories},
            indent=2, ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Dictionary":
        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            categories={k: list(v) for k, v in data.get("categories", {}).items()},
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def list_dictionaries() -> list[str]:
    """Return names of all available dictionaries (builtin + user)."""
    names = []
    for p in sorted(_HERE.glob("*.json")):
        try:
            data = json.loads(p.read_text("utf-8"))
            if data.get("_REMOVED"):
                continue
            names.append(p.stem)
        except Exception:
            continue
    for p in sorted(_USER.glob("*.json")):
        names.append(f"user/{p.stem}")
    return names


def load_dictionary(name_or_path: str | Path) -> Dictionary:
    """Load by name (e.g. 'oam') or by file path."""
    p = Path(name_or_path)
    if p.exists() and p.suffix == ".json":
        return Dictionary.from_dict(json.loads(p.read_text("utf-8")))
    for base in [_HERE, _USER]:
        candidate = base / f"{Path(str(name_or_path)).name}.json"
        if candidate.exists():
            data = json.loads(candidate.read_text("utf-8"))
            if data.get("_REMOVED"):
                raise FileNotFoundError(f"Dictionary removed: {name_or_path}")
            return Dictionary.from_dict(data)
    raise FileNotFoundError(f"Dictionary not found: {name_or_path}")


def save_user_dictionary(d: Dictionary) -> Path:
    path = _USER / f"{d.name}.json"
    d.save(path)
    return path
