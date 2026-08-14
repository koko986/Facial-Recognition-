from pydantic import BaseModel, Field


class Person(BaseModel):
    id: str
    name: str
    created_at: str
    image_count: int = 0


class RecognitionResult(BaseModel):
    predicted_person_id: str | None
    predicted_name: str
    confidence: float
    accepted: bool
    method: str


class ExperimentResult(BaseModel):
    id: str
    person_id: str | None
    person_name: str
    svd_rank: int
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    storage_reduction_percent: float
    mse: float
    psnr: float
    processing_time_ms: float
    recognition: RecognitionResult
    original_image_url: str
    compressed_image_url: str
    created_at: str


class AnalyzeResponse(BaseModel):
    original_recognition: RecognitionResult
    results: list[ExperimentResult]
    recommended_rank: int | None
    accuracy_threshold: float = Field(description="Confidence threshold used for recommendation.")


class FaceBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class RecognizeResponse(BaseModel):
    recognition: RecognitionResult
    accuracy_threshold: float
    image_url: str
    processing_time_ms: float
    face_box: FaceBox | None = None
    frame_width: int | None = None
    frame_height: int | None = None


class RegisterResponse(BaseModel):
    person: Person
    image_urls: list[str]
    message: str
