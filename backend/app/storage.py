import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import Settings


class LocalRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.data_dir
        self.images_dir = self.root / "images"
        self.people_path = self.root / "people.json"
        self.experiments_path = self.root / "experiments.json"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_json(self.people_path, [])
        self._ensure_json(self.experiments_path, [])

    def _ensure_json(self, path: Path, default: list[dict[str, Any]]) -> None:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(default, indent=2), encoding="utf-8")

    def _read(self, path: Path) -> list[dict[str, Any]]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def list_people(self) -> list[dict[str, Any]]:
        people = self._read(self.people_path)
        images = list(self.images_dir.glob("*/*.*"))
        counts: dict[str, int] = {}
        for image in images:
            counts[image.parent.name] = counts.get(image.parent.name, 0) + 1
        return [{**person, "image_count": counts.get(person["id"], 0)} for person in people]

    def create_person(self, name: str) -> dict[str, Any]:
        people = self._read(self.people_path)
        person = {
            "id": str(uuid4()),
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image_count": 0,
        }
        people.append(person)
        self._write(self.people_path, people)
        return person

    def save_image_bytes(self, data: bytes, filename: str, person_id: str | None = None) -> str:
        safe_name = Path(filename).name
        folder = self.images_dir / (person_id or "experiments")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{uuid4()}-{safe_name}"
        path.write_bytes(data)
        return path.as_posix()

    def image_paths_for_person(self, person_id: str) -> list[Path]:
        return sorted((self.images_dir / person_id).glob("*.*"))

    def save_experiment(self, experiment: dict[str, Any]) -> None:
        experiments = self._read(self.experiments_path)
        experiments.append(experiment)
        self._write(self.experiments_path, experiments)

    def list_experiments(self) -> list[dict[str, Any]]:
        return self._read(self.experiments_path)

    def public_url(self, storage_path: str) -> str:
        return storage_path
