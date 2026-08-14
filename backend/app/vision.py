import math
import time
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np


FACE_SIZE = (160, 160)


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


def recognize_face(
    probe_gray: np.ndarray,
    people: Iterable[dict],
    image_paths_for_person: Callable[[str], list[Path]],
    threshold: float,
) -> dict:
    training: list[tuple[str, str, np.ndarray]] = []
    for person in people:
        for path in image_paths_for_person(person["id"]):
            # Registration already saved preprocessed grayscale 160x160 face crops.
            # Load them directly as grayscale; do NOT re-run the face detector,
            # otherwise the descriptor becomes misaligned and even identical
            # photos will not match.
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

    if hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        person_ids = sorted({person_id for person_id, _, _ in training})
        labels = {person_id: index for index, person_id in enumerate(person_ids)}
        reverse = {
            labels[person_id]: (person_id, name)
            for person_id, name, _ in training
        }
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train([face for _, _, face in training], np.array([labels[person_id] for person_id, _, _ in training]))
        label, distance = model.predict(probe_gray)
        person_id, name = reverse[label]
        # LBPH distance: 0 is a perfect match; larger values are worse.
        # Use a more forgiving confidence mapping so identical/very similar
        # images are accepted. Distances around 0-50 indicate a strong match.
        confidence = max(0.0, min(1.0, 1.0 - (distance / 200.0)))
        return {
            "predicted_person_id": person_id if confidence >= threshold else None,
            "predicted_name": name if confidence >= threshold else "Unknown Person",
            "confidence": round(confidence, 4),
            "accepted": confidence >= threshold,
            "method": "opencv-lbph",
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