import math
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np


FACE_SIZE = (160, 160)

#: OpenCV's LBPH distance equals exactly 2 * chi-square for L1-normalized cell
#: histograms; confidence mapping below reproduces the documented 1 - d/200.
LBPH_DISTANCE_SCALE = 2.0
CONFIDENCE_SCALE = 200.0

#: Cache of extracted LBPH histograms keyed by a signature of the training set
#: (paths + mtimes + sizes). Prevents retraining the recognizer on every request.
_sample_cache: dict[str, list[dict]] = {}
_sample_cache_order: list[str] = []
_sample_cache_lock = threading.Lock()
MAX_CACHE_ENTRIES = 8


def decode_image(data: bytes) -> np.ndarray:
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image.")
    return image


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Could not encode image.")
    return buffer.tobytes()


def preprocess_face(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
    if len(faces) > 0:
        x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
        gray = gray[y : y + height, x : x + width]
    resized = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(resized)


def detect_face_region(image: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Detect the largest face in a BGR image.

    Returns (preprocessed_face_crop, (x, y, width, height)) where the bounding
    box coordinates are in the original image's pixel space. Returns (None, None)
    when no face is detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
    if len(faces) == 0:
        return None, None
    x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
    face_crop = gray[y : y + height, x : x + width]
    resized = cv2.resize(face_crop, FACE_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(resized), (int(x), int(y), int(width), int(height))


def compute_svd(gray: np.ndarray, rank: int) -> tuple[np.ndarray, float, float, float]:
    start = time.perf_counter()
    matrix = gray.astype(np.float64)
    max_rank = min(matrix.shape)
    rank = max(1, min(rank, max_rank))
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    reconstructed = (u[:, :rank] * singular_values[:rank]) @ vt[:rank, :]
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    mse = float(np.mean((matrix - reconstructed.astype(np.float64)) ** 2))
    psnr = float("inf") if mse == 0 else float(10 * math.log10((255**2) / mse))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return reconstructed, mse, psnr, elapsed_ms


def compression_stats(original_size: int, compressed_size: int) -> tuple[float, float]:
    if compressed_size <= 0:
        return 0.0, 0.0
    ratio = original_size / compressed_size
    reduction = (1 - (compressed_size / original_size)) * 100 if original_size else 0
    return round(ratio, 4), round(reduction, 2)


def histogram_embedding(gray: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
    return hist / (np.linalg.norm(hist) + 1e-9)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def _lbph_histogram(gray: np.ndarray) -> np.ndarray:
    """Extract the LBPH cell histogram OpenCV uses internally (L1-normalized per cell)."""
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train([gray], np.array([0], dtype=np.int32))
    return model.getHistograms()[0].reshape(-1)


def _chi_square_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, ((a - b) ** 2) / denom, 0.0)
    return float(np.sum(ratio))


def _training_signature(people: Iterable[dict], image_paths_for_person: Callable[[str], list[Path]]) -> str:
    parts: list[str] = []
    for person in people:
        parts.append(person["id"])
        parts.append(person["name"])
        for path in image_paths_for_person(person["id"]):
            try:
                stat = path.stat()
                parts.append(f"{path}:{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                parts.append(f"{path}:missing")
    return "|".join(parts)


def _load_training_samples(people: Iterable[dict], image_paths_for_person: Callable[[str], list[Path]]) -> list[dict]:
    """Load preprocessed training faces once and cache their LBPH histograms.

    Training images were saved as preprocessed 160x160 grayscale crops at
    registration time; they are read directly, never re-detected.
    """
    signature = _training_signature(people, image_paths_for_person)
    with _sample_cache_lock:
        cached = _sample_cache.get(signature)
        if cached is not None:
            return cached

    samples: list[dict] = []
    for person in people:
        for path in image_paths_for_person(person["id"]):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            if gray.shape[:2] != FACE_SIZE:
                gray = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
            samples.append(
                {
                    "person_id": person["id"],
                    "name": person["name"],
                    "hist": _lbph_histogram(gray),
                }
            )

    with _sample_cache_lock:
        _sample_cache[signature] = samples
        _sample_cache_order.append(signature)
        while len(_sample_cache_order) > MAX_CACHE_ENTRIES:
            oldest = _sample_cache_order.pop(0)
            _sample_cache.pop(oldest, None)
    return samples


def recognize_face(
    probe_gray: np.ndarray,
    people: Iterable[dict],
    image_paths_for_person: Callable[[str], list[Path]],
    threshold: float,
    require_separation: bool = False,
) -> dict:
    if not people:
        return {
            "predicted_person_id": None,
            "predicted_name": "No registered faces",
            "confidence": 0.0,
            "accepted": False,
            "method": "histogram-fallback",
        }

    if hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        samples = _load_training_samples(people, image_paths_for_person)
        if not samples:
            return {
                "predicted_person_id": None,
                "predicted_name": "No registered faces",
                "confidence": 0.0,
                "accepted": False,
                "method": "histogram-fallback",
            }

        probe_hist = _lbph_histogram(probe_gray)

        # Distance per registered person = min distance over their training samples.
        # LBPH_DISTANCE_SCALE reproduces OpenCV's LBPHFaceRecognizer.predict exactly.
        best_per_person: dict[str, dict] = {}
        for sample in samples:
            distance = LBPH_DISTANCE_SCALE * _chi_square_distance(probe_hist, sample["hist"])
            current = best_per_person.get(sample["person_id"])
            if current is None or distance < current["distance"]:
                best_per_person[sample["person_id"]] = {
                    "person_id": sample["person_id"],
                    "name": sample["name"],
                    "distance": distance,
                }
        ranked = sorted(best_per_person.values(), key=lambda row: row["distance"])
        best = ranked[0]
        confidence = max(0.0, min(1.0, 1.0 - (best["distance"] / CONFIDENCE_SCALE)))

        separated = True
        if require_separation and len(ranked) >= 2:
            # The closest person must be clearly closer than the runner-up,
            # otherwise an unknown face would be accepted on raw distance alone.
            separated = best["distance"] <= ranked[1]["distance"] * 0.85

        accepted = confidence >= threshold and separated
        person_id = best["person_id"] if accepted else None
        return {
            "predicted_person_id": person_id,
            "predicted_name": best["name"] if accepted else "Unknown Person",
            "confidence": round(confidence, 4),
            "accepted": accepted,
            "method": "opencv-lbph",
        }

    training: list[tuple[str, str, np.ndarray]] = []
    for person in people:
        for path in image_paths_for_person(person["id"]):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            if gray.shape[:2] != FACE_SIZE:
                gray = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
            training.append((person["id"], person["name"], gray))

    if not training:
        return {
            "predicted_person_id": None,
            "predicted_name": "No registered faces",
            "confidence": 0.0,
            "accepted": False,
            "method": "histogram-fallback",
        }

    probe = histogram_embedding(probe_gray)
    person_id, name, confidence = max(
        ((person_id, name, cosine_similarity(probe, histogram_embedding(face))) for person_id, name, face in training),
        key=lambda row: row[2],
    )
    return {
        "predicted_person_id": person_id if confidence >= threshold else None,
        "predicted_name": name if confidence >= threshold else "Unknown Person",
        "confidence": round(confidence, 4),
        "accepted": confidence >= threshold,
        "method": "histogram-fallback",
    }