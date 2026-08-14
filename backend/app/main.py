from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .models import AnalyzeResponse, ExperimentResult, Person, RecognitionResult, RegisterResponse
from .sqlite_store import Repository
from .vision import compression_stats, compute_svd, decode_image, encode_jpeg, preprocess_face, recognize_face

settings = get_settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SVD FaceVault API", version="0.1.0")
app.mount("/data", StaticFiles(directory=settings.data_dir), name="data")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_repository(settings: Settings = Depends(get_settings)) -> Repository:
    return Repository(settings)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name, "database": "sqlite"}


@app.get("/api/people", response_model=list[Person])
def list_people(repository: Repository = Depends(get_repository)) -> list[dict]:
    return repository.list_people()


@app.get("/api/experiments", response_model=list[ExperimentResult])
def list_experiments(repository: Repository = Depends(get_repository)) -> list[dict]:
    return repository.list_experiments()


@app.post("/api/register", response_model=RegisterResponse)
async def register_person(
    name: str = Form(...),
    images: list[UploadFile] = File(...),
    repository: Repository = Depends(get_repository),
) -> RegisterResponse:
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name is required.")
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    person = repository.create_person(name.strip())
    urls: list[str] = []
    for image in images:
        data = await image.read()
        try:
            face = preprocess_face(decode_image(data))
            encoded = encode_jpeg(face)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        path = repository.save_image_bytes(encoded, image.filename or "face.jpg", person["id"], "training")
        urls.append(repository.public_url(path))

    person["image_count"] = len(urls)
    return RegisterResponse(person=Person(**person), image_urls=urls, message="Person registered successfully.")


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    image: UploadFile = File(...),
    repository: Repository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
) -> AnalyzeResponse:
    data = await image.read()
    try:
        face = preprocess_face(decode_image(data))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    people = repository.list_people()
    original_recognition = recognize_face(face, people, repository.image_paths_for_person, settings.recognition_threshold)
    original_bytes = encode_jpeg(face, quality=92)
    original_path = repository.save_image_bytes(original_bytes, image.filename or "original.jpg", image_kind="original")
    original_url = repository.public_url(original_path)

    results: list[dict] = []
    for rank in settings.rank_values:
        compressed_face, mse, psnr, elapsed_ms = compute_svd(face, rank)
        compressed_bytes = encode_jpeg(compressed_face, quality=85)
        compressed_path = repository.save_image_bytes(
            compressed_bytes,
            f"rank-{rank}.jpg",
            image_kind="compressed",
            svd_rank=rank,
        )
        compression_ratio, storage_reduction = compression_stats(len(original_bytes), len(compressed_bytes))
        recognition = recognize_face(compressed_face, people, repository.image_paths_for_person, settings.recognition_threshold)
        experiment = {
            "id": str(uuid4()),
            "person_id": original_recognition["predicted_person_id"],
            "person_name": original_recognition["predicted_name"],
            "svd_rank": rank,
            "original_size_bytes": len(original_bytes),
            "compressed_size_bytes": len(compressed_bytes),
            "compression_ratio": compression_ratio,
            "storage_reduction_percent": storage_reduction,
            "mse": round(mse, 4),
            "psnr": round(psnr, 4) if psnr != float("inf") else psnr,
            "processing_time_ms": round(elapsed_ms, 2),
            "recognition": recognition,
            "original_image_url": original_url,
            "compressed_image_url": repository.public_url(compressed_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        repository.save_experiment(experiment)
        results.append(experiment)

    recommended = next(
        (
            result["svd_rank"]
            for result in sorted(results, key=lambda row: row["svd_rank"])
            if result["recognition"]["accepted"]
            and result["recognition"]["predicted_person_id"] == original_recognition["predicted_person_id"]
        ),
        None,
    )

    return AnalyzeResponse(
        original_recognition=RecognitionResult(**original_recognition),
        results=[ExperimentResult(**result) for result in results],
        recommended_rank=recommended,
        accuracy_threshold=settings.recognition_threshold,
    )
