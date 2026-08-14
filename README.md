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

## Theory — The Mathematics Behind SVD FaceVault

### 1. Singular Value Decomposition (SVD)

The core of this project. Any real $m \times n$ matrix $A$ can be **factorized** into three matrices:

$$
A = U \Sigma V^T
$$

Where:

- $U$ — an $m \times m$ **orthogonal** matrix (left singular vectors) representing the "left" structure of the image (rows).
- $\Sigma$ — an $m \times n$ **diagonal** matrix whose diagonal entries $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0$ are called the **singular values**. They rank the amount of "energy" (variance) carried by each component.
- $V^T$ — an $n \times n$ **orthogonal** matrix (right singular vectors) representing the "right" structure of the image (columns).

In Python/NumPy: `u, s, vt = np.linalg.svd(A, full_matrices=False)`.

#### Rank-$k$ Approximation (Compression)

To compress an image we keep only the first $k$ singular values and vectors:

$$
A_k = U_k \, \Sigma_k \, V_k^T = \sum_{i=1}^{k} \sigma_i \, u_i \, v_i^T
$$

where $U_k$ is the first $k$ columns of $U$ and $V_k^T$ is the first $k$ rows of $V^T$.

This is known as the **Eckart–Young theorem**: $A_k$ is the best rank-$k$ approximation to $A$ in the Frobenius norm:

$$
A_k = \arg\min_{\text{rank}(B) \le k} \| A - B \|_F
$$

**Storage cost**: instead of storing $m \times n$ pixel values, we store only:

$$
\text{stored values} = k(m + n + 1)
$$

For a $160 \times 160$ face image:
- Full matrix: $160 \times 160 = 25{,}600$ values
- Rank 10: $10 \times (160 + 160 + 1) = 3{,}210$ values — over **87% smaller**

This is why low-rank reconstruction requires dramatically less storage.

In the code (`compute_svd`):

```python
u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
reconstructed = (u[:, :rank] * singular_values[:rank]) @ vt[:rank, :]
```

---

### 2. Mean Squared Error (MSE)

MSE measures the average squared difference between the original image $A$ and the reconstructed image $A_k$:

$$
\text{MSE} = \frac{1}{mn} \sum_{i=1}^{m} \sum_{j=1}^{n} \left( A(i,j) - A_k(i,j) \right)^2
$$

- $\text{MSE} = 0$ → perfect reconstruction (lossless).
- Larger MSE → more distortion from aggressive compression.

In the code:

```python
mse = float(np.mean((matrix - reconstructed.astype(np.float64)) ** 2))
```

---

### 3. Peak Signal-to-Noise Ratio (PSNR)

PSNR expresses the reconstruction quality in **decibels (dB)**, using the maximum possible pixel value $L$ (255 for 8-bit grayscale images):

$$
\text{PSNR} = 10 \cdot \log_{10} \left( \frac{L^2}{\text{MSE}} \right)
$$

Where $L = 255$.

Interpretation:
- **PSNR → ∞ dB** when MSE = 0 (perfect).
- **PSNR ≈ 30–40 dB** → very good quality.
- **PSNR < 20 dB** → visibly degraded.

The code:

```python
psnr = float("inf") if mse == 0 else float(10 * math.log10((255**2) / mse))
```

---

### 4. Compression Ratio & Storage Reduction

The **compression ratio** compares original file size vs compressed file size:

$$
\text{Compression Ratio} = \frac{S_{original}}{S_{compressed}}
$$

The **storage reduction percentage** shows how much space was saved:

$$
\text{Storage Reduction} = \left( 1 - \frac{S_{compressed}}{S_{original}} \right) \times 100\%
$$

Example: if the original JPEG is 100 KB and the rank-10 reconstruction is 25 KB, then:

- Compression Ratio = $100 / 25 = 4{:}1$
- Storage Reduction = $75\%$

The code:

```python
ratio = original_size / compressed_size
reduction = (1 - (compressed_size / original_size)) * 100 if original_size else 0
```

---

### 5. Face Recognition — Local Binary Patterns Histograms (LBPH)

LBPH is a texture-based face descriptor used by OpenCV (`cv2.face.LBPHFaceRecognizer`).

#### Step 1 — Local Binary Pattern

For each pixel $P_c$, compare it with its 8 neighbours $P_0, \dots, P_7$ in a $3 \times 3$ window:

$$
\text{LBP}(x_c, y_c) = \sum_{i=0}^{7} s(P_i - P_c) \cdot 2^i,
\qquad
s(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{otherwise} \end{cases}
$$

This produces an 8-bit binary code (0–255) per pixel, capturing local texture patterns invariant to illumination changes.

#### Step 2 — Histogram

The LBP codes are collected into a histogram of 256 bins per face region, normalised to unit length.

#### Step 3 — Distance & Confidence

LBPH compares a probe face against the trained model using the **Chi-square distance**:

$$
d(H_1, H_2) = \sum_i \frac{(H_1(i) - H_2(i))^2}{H_1(i) + H_2(i)}
$$

A **lower distance** means a better match (distance $0$ = identical images). The distance is converted to a confidence score in $[0, 1]$:

$$
\text{Confidence} = \max\left(0,\; \min\left(1,\; 1 - \frac{d}{200}\right)\right)
$$

The prediction is **accepted** only when:

$$
\text{Confidence} \ge \text{threshold} \quad (\text{default } 0.63)
$$

Two practical notes for the live camera path (`/api/recognize`):

- Webcam frames are noisier than uploaded stills, so live recognition uses a slightly more forgiving `LIVE_RECOGNITION_THRESHOLD` (default `0.55`). The analysis pipeline (`/api/analyze`) keeps the stricter `0.63` for consistent SVD experiments.
- When more than one participant is registered, live recognition also requires the closest match to be clearly separated from the runner-up (at least 15% closer), so an unknown face is not accepted on raw distance alone.
- Training-face LBPH histograms are extracted once and cached (keyed by file path/mtime/size), so each live frame only costs one histogram extraction and a distance sweep — no retraining per frame.

---

### 6. Histogram Fallback — Cosine Similarity

If the LBPH module is unavailable, the app falls back to a global intensity-histogram embedding and **cosine similarity**:

$$
\text{cos}(\theta) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \, \|\vec{b}\|} = \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2} \cdot \sqrt{\sum_i b_i^2}}
$$

- $\cos(\theta) = 1$ → identical histograms.
- $\cos(\theta) \to 0$ → unrelated images.

---

### Why This Project Matters

1. **Data reduction**: SVD shows that faces can be represented by far fewer numbers with minimal recognition loss.
2. **Rank–quality trade-off**: lower ranks save storage but increase MSE and lower PSNR — the project visualizes exactly where recognition starts to fail.
3. **Recommended rank**: the app finds the *lowest* rank $k$ where recognition is still accepted, giving the best compression without sacrificing accuracy.
4. **Applied linear algebra**: connects abstract concepts ($U \Sigma V^T$, eigenvalues/variance) to a tangible, visual real-world application.

## Research Demo Flow

1. Register consenting participants with several face images each.
2. Capture or upload a test face image.
3. Run recognition on the original face crop.
4. Generate SVD reconstructions at ranks `5, 10, 20, 30, 50, 100`.
5. Compare confidence, storage reduction, MSE, PSNR, and processing time.
6. Present the lowest rank that still keeps recognition accepted.

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

## Tests

```bash
cd backend
pytest