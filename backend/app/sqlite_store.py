import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import Settings


class Repository:
    """SQLite metadata store plus local image files."""

    def __init__(self, settings: Settings) -> None:
        self.root = settings.data_dir
        self.images_dir = self.root / "images"
        self.db_path = settings.sqlite_path
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma foreign_keys = on")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                create table if not exists people (
                  id text primary key,
                  name text not null,
                  created_at text not null
                );

                create table if not exists face_images (
                  id text primary key,
                  person_id text references people(id) on delete cascade,
                  storage_path text not null,
                  image_kind text not null check (image_kind in ('training', 'original', 'compressed')),
                  svd_rank integer,
                  created_at text not null
                );

                create table if not exists compression_experiments (
                  id text primary key,
                  person_id text references people(id) on delete set null,
                  person_name text not null,
                  svd_rank integer not null,
                  original_size_bytes integer not null,
                  compressed_size_bytes integer not null,
                  compression_ratio real not null,
                  storage_reduction_percent real not null,
                  mse real not null,
                  psnr real not null,
                  processing_time_ms real not null,
                  predicted_person_id text references people(id) on delete set null,
                  predicted_name text not null,
                  confidence real not null,
                  accepted integer not null,
                  recognition_method text not null,
                  original_image_path text not null,
                  compressed_image_path text not null,
                  created_at text not null
                );
                """
            )

    def list_people(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select p.id, p.name, p.created_at, count(fi.id) as image_count
                from people p
                left join face_images fi on fi.person_id = p.id and fi.image_kind = 'training'
                group by p.id
                order by p.created_at
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def create_person(self, name: str) -> dict[str, Any]:
        person = {
            "id": str(uuid4()),
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image_count": 0,
        }
        with self._connect() as connection:
            connection.execute(
                "insert into people (id, name, created_at) values (?, ?, ?)",
                (person["id"], person["name"], person["created_at"]),
            )
        return person

    def save_image_bytes(
        self,
        data: bytes,
        filename: str,
        person_id: str | None = None,
        image_kind: str = "training",
        svd_rank: int | None = None,
    ) -> str:
        safe_name = Path(filename).name
        folder = self.images_dir / (person_id or "experiments")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{uuid4()}-{safe_name}"
        path.write_bytes(data)
        storage_path = path.as_posix()

        with self._connect() as connection:
            connection.execute(
                """
                insert into face_images (id, person_id, storage_path, image_kind, svd_rank, created_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    person_id,
                    storage_path,
                    image_kind,
                    svd_rank,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return storage_path

    def image_paths_for_person(self, person_id: str) -> list[Path]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select storage_path
                from face_images
                where person_id = ? and image_kind = 'training'
                order by created_at
                """,
                (person_id,),
            ).fetchall()
        return [Path(row["storage_path"]) for row in rows]

    def save_experiment(self, experiment: dict[str, Any]) -> None:
        recognition = experiment["recognition"]
        with self._connect() as connection:
            connection.execute(
                """
                insert into compression_experiments (
                  id, person_id, person_name, svd_rank, original_size_bytes, compressed_size_bytes,
                  compression_ratio, storage_reduction_percent, mse, psnr, processing_time_ms,
                  predicted_person_id, predicted_name, confidence, accepted, recognition_method,
                  original_image_path, compressed_image_path, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment["id"],
                    experiment["person_id"],
                    experiment["person_name"],
                    experiment["svd_rank"],
                    experiment["original_size_bytes"],
                    experiment["compressed_size_bytes"],
                    experiment["compression_ratio"],
                    experiment["storage_reduction_percent"],
                    experiment["mse"],
                    experiment["psnr"],
                    experiment["processing_time_ms"],
                    recognition["predicted_person_id"],
                    recognition["predicted_name"],
                    recognition["confidence"],
                    1 if recognition["accepted"] else 0,
                    recognition["method"],
                    experiment["original_image_url"],
                    experiment["compressed_image_url"],
                    experiment["created_at"],
                ),
            )

    def list_experiments(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select *
                from compression_experiments
                order by created_at
                limit 200
                """
            ).fetchall()
        return [
            {
                "id": row["id"],
                "person_id": row["person_id"],
                "person_name": row["person_name"],
                "svd_rank": row["svd_rank"],
                "original_size_bytes": row["original_size_bytes"],
                "compressed_size_bytes": row["compressed_size_bytes"],
                "compression_ratio": row["compression_ratio"],
                "storage_reduction_percent": row["storage_reduction_percent"],
                "mse": row["mse"],
                "psnr": row["psnr"],
                "processing_time_ms": row["processing_time_ms"],
                "recognition": {
                    "predicted_person_id": row["predicted_person_id"],
                    "predicted_name": row["predicted_name"],
                    "confidence": row["confidence"],
                    "accepted": bool(row["accepted"]),
                    "method": row["recognition_method"],
                },
                "original_image_url": row["original_image_path"],
                "compressed_image_url": row["compressed_image_path"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def public_url(self, storage_path: str) -> str:
        return storage_path
