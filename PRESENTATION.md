# SVD FaceVault — Presentation Guide (Math + Explanations)

A slide-by-slide deck built around the mathematics of the project. Every equation
is followed by a plain-language explanation, a "why it matters" note, and where it
lives in the code. All measured numbers below come from real runs of the app.

The one-number summary of the whole project:

> **Rank-10 SVD keeps the face recognizable while storing 43% less data.
> Rank-5 stores more but recognition fails. SVD has a sweet spot.**

---

## 0. The demo flow (2 minutes)

1. Register a participant with several face photos.
2. Point the webcam at someone — the LBPH matcher says who it is (or "Unknown Person").
3. Upload/capture a test face → the app rebuilds it at ranks $5, 10, 20, 30, 50, 100$.
4. For each rank it measures MSE, PSNR, storage saved, and **recognition confidence**.
5. It reports the *lowest* rank where recognition still succeeds = the recommended rank.

The mathematics answers one question: **how few numbers can we store a face with while
still recognizing it?**

---

## 1. Singular Value Decomposition (SVD) — the engine

Any real $m \times n$ matrix $A$ (each face crop is a $160 \times 160$ pixel matrix) can be factored as

$$
\boxed{A = U\,\Sigma\,V^T}
$$

with

- $U$ — $m \times m$ **orthogonal** matrix: columns $u_1, \dots, u_m$ are the *left singular vectors* (basis for the row picture; the "directions" among image rows).
- $\Sigma$ — $m \times n$ **diagonal** matrix with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0$, the **singular values** (sorted by importance).
- $V^T$ — $n \times n$ **orthogonal** matrix: rows are the *right singular vectors* $v_1, \dots, v_n$.

**Plain English:** SVD rotates the image into a coordinate system where the axes are
ordered by how much "energy" (variance) each direction carries. $\sigma_i$ tells you how
important axis $i$ is.

**Why it matters:** typical faces differ along a few strong directions (eyes, jaw,
lighting). Most singular values are tiny, so most of a face is redundant.

**Where in the code:**
```python
u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)   # vision.py
```

### Connection to eigenvalues (for the linear-algebra tie-in)

The singular values are the square roots of the eigenvalues of $A^TA$ (and $AA^T$):

$$
\sigma_i^2 = \lambda_i(A^TA), \qquad \sigma_i = \sqrt{\lambda_i(A^TA)}
$$

$A^TA$ is symmetric positive semi-definite, so its eigenvalues are real and non-negative —
this is why SVD exists for *every* matrix, even non-square ones. The left/right singular
vectors are the eigenvectors of $AA^T$ / $A^TA$.

---

## 2. Rank-$k$ Approximation — the compression step

To compress, keep only the $k$ largest singular values and their vectors:

$$
\boxed{A_k = U_k \,\Sigma_k \, V_k^T = \sum_{i=1}^{k} \sigma_i \, u_i \, v_i^T}
$$

**Eckart–Young theorem:** $A_k$ is the best rank-$k$ approximation to $A$ in the
Frobenius norm — no other rank-$k$ matrix comes closer:

$$
\|A - A_k\|_F = \min_{\text{rank}(B) \le k} \|A - B\|_F
    = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}
$$

**Plain English:** the approximation error is exactly the "energy" in the singular values
you threw away. Big $\sigma_{k+1}$ → visible damage; tiny ones → invisible damage.

**Why it matters:** you control quality with one integer, $k$ — and the math tells you
exactly how much error you buy at that $k$.

**Where in the code:**
```python
reconstructed = (u[:, :rank] * singular_values[:rank]) @ vt[:rank, :]   # vision.py
```

### Energy retention (how much "information" is kept)

Define the total energy as the sum of squared singular values:

$$
E_k = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{n} \sigma_i^2} \times 100\%
$$

Faces collapse fast: a $160 \times 160$ face keeps >95% of its energy by rank ~30
for typical photos — the visual table below shows the consequences.

---

## 3. Storage analysis — how much we save

Instead of storing $m \times n = 25600$ pixel values, a rank-$k$ SVD stores
$k$ singular values + $k$ column vectors + $k$ row vectors:

$$
\boxed{\text{stored values} = k\,(m + n + 1)}
$$

| rank $k$ | values stored | vs $25{,}600$ | |
|---|---|---|---|
| full | 25,600 | 100% | |
| 100 | 32,100 | *larger* | (no saving) |
| 30 | 9,630 | 62% smaller | |
| 10 | 3,210 | **87% smaller** | |
| 5 | 1,605 | 94% smaller | |

**Plain English:** storage cost is linear in $k$ — roughly $k \times 321$ values for a
$160 \times 160$ face — instead of quadratic. That is the entire compression story.

Measured (from the app, real runs — JPEG file sizes after re-encoding):

| rank | storage saved (measured) | MSE | PSNR | confidence | accepted |
|---|---|---|---|---|---|
| 100 | 27% | 0.85 | 48.8 dB | 0.75 | ✅ |
| 50 | 29% | 10.3 | 38.0 dB | 0.76 | ✅ |
| 30 | 32% | 34.8 | 32.7 dB | 0.75 | ✅ |
| 20 | 36% | 75.2 | 29.4 dB | 0.72 | ✅ |
| **10** | **43%** | 252.8 | 24.1 dB | 0.65 | ✅ ← recommended |
| 5 | 50% | 650.5 | 20.0 dB | 0.58 | ❌ recognition fails |

> **The punch line of the demo:** rank 10 saves 43% of storage and the face is *still*
> recognized. Rank 5 saves only 7% more — and the recognizer no longer believes the
> face is real.

---

## 4. Quality metrics — MSE and PSNR

### Mean Squared Error

$$
\boxed{\text{MSE} = \frac{1}{mn}\sum_{i=1}^{m}\sum_{j=1}^{n}\big(A(i,j) - A_k(i,j)\big)^2}
$$

- 0 = lossless; in this project $\text{MSE}_5 = 650$ vs $\text{MSE}_{100} = 0.85$.
- Note how MSE explodes for low ranks — it is exactly $\frac{1}{mn}\|A - A_k\|_F^2$, so
  it connects directly to the Eckart–Young error formula.

### Peak Signal-to-Noise Ratio

$$
\boxed{\text{PSNR} = 10\log_{10}\!\left(\frac{L^2}{\text{MSE}}\right),\quad L = 255}
$$

Equivalent form that shows the structure: $\text{PSNR} = 20\log_{10}(255) - 10\log_{10}(\text{MSE}) \approx 48.13 - 10\log_{10}(\text{MSE})$.

| PSNR | verdict |
|---|---|
| → ∞ | perfect |
| 35–50 dB | excellent (rank 30–100) |
| 24–30 dB | visibly degraded but usable (rank 10–20) |
| < 20 dB | badly damaged (rank 5 fails recognition) |

**Where in the code:**
```python
psnr = float("inf") if mse == 0 else float(10 * math.log10((255**2) / mse))
```

### Compression ratio & storage reduction

$$
\text{ratio} = \frac{S_{\text{original}}}{S_{\text{compressed}}},
\qquad
\text{reduction} = \left(1 - \frac{S_{\text{compressed}}}{S_{\text{original}}}\right)\times 100\%
$$

---

## 5. Face recognition — Local Binary Patterns Histograms (LBPH)

### Step 1 — the Local Binary Pattern code

For each center pixel $P_c$, compare the 8 neighbours $P_0,\dots,P_7$:

$$
\text{LBP}(x_c,y_c) = \sum_{i=0}^{7} s(P_i - P_c)\,2^i,
\qquad
s(z) = \begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}
$$

Each $3 \times 3$ window becomes one number in $[0, 255]$ that describes the local
texture. The binary code is **illumination-invariant**: scaling all pixel values by a
constant changes the comparisons little, so it survives lighting changes.

### Step 2 — histograms

For every $20 \times 20$ cell of the $160 \times 160$ face (an $8 \times 8$ grid), count
the LBP codes into a 256-bin histogram, normalized per cell; concatenate the 64 cells:

$$
H = \big[\,h_{\text{cell}_1},\; h_{\text{cell}_2},\; \dots,\; h_{\text{cell}_{64}}\,\big]
$$

### Step 3 — distance and confidence

Compare probe histogram $H_1$ against training histogram $H_2$ with **chi-square**:

$$
\boxed{d(H_1,H_2) = \sum_i \frac{\big(H_1(i) - H_2(i)\big)^2}{H_1(i) + H_2(i)}}
$$

Lower distance = better match ($d=0$ identical). Map distance to a confidence in $[0,1]$:

$$
\text{confidence} = \max\!\left(0,\;1 - \frac{d}{200}\right)
$$

Accept only when confidence ≥ threshold (analysis `0.63`; live webcam `0.55` — see §6).

> Note: OpenCV's LBPH distance equals exactly $2 \times$ the chi-square above; the app
> reproduces it (`LBPH_DISTANCE_SCALE = 2.0` in `vision.py`) and caches training
> histograms, so each live frame costs ~3 ms.

**Why it matters for the demo:** the *same* math that compresses the image also has to
recognize it. As $k$ drops, distortion grows and $d$ grows — confidence 0.75 → 0.65 → 0.58 —
exactly the trend in the table in §3.

---

## 6. The live camera path

- Every 1.2 s the frontend grabs a frame and calls `/api/recognize`.
- Backend detects the face (Haar cascade), extracts the probe LBPH histogram, and finds
  the registered person with the **minimum chi-square distance** across **all** of that
  person's training samples.
- Webcam frames are noisier than uploaded stills → live threshold `0.55`.
- When several people are registered, a second condition applies: the best match must be
  clearly separated from the runner-up,
  $d_{\text{best}} \le 0.85 \cdot d_{\text{runner-up}}$ — so an unknown face close to
  two registered people is still labeled **Unknown Person**, not misidentified.

---

## 7. What the charts on screen mean

- **Confidence vs rank** — the recognition curve: flat until the face falls apart.
- **Storage saved vs rank** — monotone growth; the payoff is front-loaded.
- **PSNR vs rank** — quality falls smoothly; recognition is what breaks first.

The app then answers: *"the lowest rank at which recognition stayed accepted"* →
`recommended_rank`. In the measured run that is **rank 10 — 43% smaller, still "ako".**

---

## 8. Symbols cheat-sheet (handy for Q&A)

| symbol | meaning |
|---|---|
| $A$, $A_k$ | face image matrix, its rank-$k$ approximation |
| $U$, $V$ | orthogonal matrices (left/right singular vectors) |
| $\Sigma$, $\sigma_i$ | diagonal matrix of singular values; the $i$-th one |
| $k$ | rank (compression level) |
| $m, n$ | image height, width (both 160 here) |
| MSE, PSNR | squared error; peak signal-to-noise ratio in dB |
| $H_1, H_2$ | LBPH histograms (probe, training) |
| $d$ | chi-square distance between histograms |
| $L$ | peak pixel value = 255 |

## 9. Talking points for the final slide

1. SVD = "find the axes of importance and keep the top $k$".
2. Error is *provably minimal* for a given $k$ (Eckart–Young) — $\|A - A_k\|_F = \sqrt{\sum_{i>k}\sigma_i^2}$.
3. Storage is linear in $k$ instead of quadratic: $k(m+n+1)$ vs $mn$.
4. Quality (PSNR) degrades smoothly, but **recognition confidence collapses sharply
   below the recommended rank** — the interesting failure is in the recognizer, not the PSNR.
5. Demonstrated: rank 10 → 43% storage saved, recognition still accepted; rank 5 → fails.
6. Future work: more training frames per person, per-cell separation margins, deeper
   embeddings (e.g., face-nets) replacing LBPH histograms — the SVD story stays identical.