# SVD FaceVault

SVD FaceVault is a university-friendly research demo for studying how Singular Value Decomposition image compression affects facial recognition. It uses a React dashboard, a FastAPI computer-vision backend, and a local SQLite database.

## What is included

- React + Vite + TypeScript dashboard with registration, webcam/upload analysis, result charts, and rank previews.
- FastAPI backend for face preprocessing, OpenCV LBPH recognition when available, SVD compression, MSE, PSNR, compression ratio, and timing metrics.
- SQLite database for participants, image metadata, and compression experiment results.
- Local image file storage under `backend/data/images`, with image paths stored in SQLite.

## Project Structure

```text
backend/
  app/              FastAPI API, CV, SVD, SQLite persistence
  tests/            SVD metric tests
frontend/
  src/              React dashboard
database/
  schema.sql        SQLite schema reference
```

## Backend Setup

Use Python 3.11 or 3.12. Python 3.14 may try to build NumPy/OpenCV from source on Windows.

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000`. The SQLite database is created automatically at `backend/data/facevault.db`.

## Frontend Setup

```bash
cd frontend
copy .env.example .env
npm install
npm run dev
```

The app runs at `http://localhost:5173`.

On Windows PowerShell, use `npm.cmd run dev` if script execution blocks `npm`.

## Database

No cloud database is required. The backend uses Python's built-in SQLite support, so there are no database credentials, accounts, buckets, or hosted services to configure.

The schema is documented in `database/schema.sql`, and the app creates the tables automatically on startup.

## Research Demo Flow

1. Register consenting participants with several face images each.
2. Capture or upload a test face image.
3. Run recognition on the original face crop.
4. Generate SVD reconstructions at ranks `5, 10, 20, 30, 50, 100`.
5. Compare confidence, storage reduction, MSE, PSNR, and processing time.
6. Present the lowest rank that still keeps recognition accepted.

## Tests

```bash
cd backend
pytest
```
