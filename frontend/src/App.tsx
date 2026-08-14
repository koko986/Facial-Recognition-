import { useEffect, useMemo, useRef, useState } from "react";
import { BarChart3, Camera, CircleUserRound, FlaskConical, RefreshCcw, Upload } from "lucide-react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { analyzeImage, fetchExperiments, fetchPeople, registerPerson, resolveMediaUrl } from "./api";
import type { AnalyzeResponse, ExperimentResult, Person } from "./types";

type Tab = "register" | "capture" | "results";

export function App() {
  const [tab, setTab] = useState<Tab>("capture");
  const [people, setPeople] = useState<Person[]>([]);
  const [experiments, setExperiments] = useState<ExperimentResult[]>([]);
  const [latest, setLatest] = useState<AnalyzeResponse | null>(null);
  const [name, setName] = useState("");
  const [registerFiles, setRegisterFiles] = useState<File[]>([]);
  const [analysisFile, setAnalysisFile] = useState<File | null>(null);
  const [status, setStatus] = useState("Ready");
  const [isBusy, setIsBusy] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const refresh = async () => {
    const [peopleRows, experimentRows] = await Promise.all([fetchPeople(), fetchExperiments()]);
    setPeople(peopleRows);
    setExperiments(experimentRows.slice(-100));
  };

  useEffect(() => {
    refresh().catch((error) => setStatus(error.message));
  }, []);

  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 720 }, audio: false });
    if (videoRef.current) videoRef.current.srcObject = stream;
  };

  const captureFrame = async () => {
    if (!videoRef.current || !canvasRef.current) return null;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 960;
    canvas.height = video.videoHeight || 720;
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    return blob ? new File([blob], "webcam-capture.jpg", { type: "image/jpeg" }) : null;
  };

  const runAnalysis = async (file: File | null) => {
    if (!file) {
      setStatus("Choose an image or capture from the camera first.");
      return;
    }
    setIsBusy(true);
    setStatus("Running face recognition and SVD experiment...");
    try {
      const response = await analyzeImage(file);
      setLatest(response);
      await refresh();
      setTab("results");
      setStatus("Experiment complete.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Analysis failed.");
    } finally {
      setIsBusy(false);
    }
  };

  const submitRegistration = async () => {
    if (!name.trim() || registerFiles.length === 0) {
      setStatus("Add a name and at least one face image.");
      return;
    }
    setIsBusy(true);
    setStatus("Registering participant...");
    try {
      await registerPerson(name, registerFiles);
      setName("");
      setRegisterFiles([]);
      await refresh();
      setStatus("Participant registered.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Registration failed.");
    } finally {
      setIsBusy(false);
    }
  };

  const chartRows = useMemo(() => latest?.results ?? experiments.slice(-12), [latest, experiments]);
  const acceptedRows = chartRows.filter((row) => row.recognition.accepted);
  const bestRank = latest?.recommended_rank ?? acceptedRows.sort((a, b) => a.svd_rank - b.svd_rank)[0]?.svd_rank ?? null;

  return (
    <main>
      <aside className="sidebar">
        <div>
          <p className="eyebrow">Research Demo</p>
          <h1>SVD FaceVault</h1>
        </div>
        <nav>
          <button className={tab === "register" ? "active" : ""} onClick={() => setTab("register")} title="Register">
            <CircleUserRound size={20} /> Register
          </button>
          <button className={tab === "capture" ? "active" : ""} onClick={() => setTab("capture")} title="Capture">
            <Camera size={20} /> Capture
          </button>
          <button className={tab === "results" ? "active" : ""} onClick={() => setTab("results")} title="Results">
            <BarChart3 size={20} /> Results
          </button>
        </nav>
        <div className="stat">
          <span>Participants</span>
          <strong>{people.length}</strong>
        </div>
        <div className="stat">
          <span>Experiments</span>
          <strong>{experiments.length}</strong>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Current status</p>
            <strong>{status}</strong>
          </div>
          <button className="iconButton" onClick={() => refresh()} title="Refresh data">
            <RefreshCcw size={18} />
          </button>
        </header>

        {tab === "register" && (
          <section className="panel">
            <div className="sectionTitle">
              <CircleUserRound size={22} />
              <h2>Participant Registration</h2>
            </div>
            <div className="formGrid">
              <label>
                Participant name
                <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Alice" />
              </label>
              <label>
                Training images
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={(event) => setRegisterFiles(Array.from(event.target.files ?? []))}
                />
              </label>
            </div>
            <button disabled={isBusy} onClick={submitRegistration}>
              <Upload size={18} /> Register Face Set
            </button>
            <div className="peopleGrid">
              {people.map((person) => (
                <article className="person" key={person.id}>
                  <strong>{person.name}</strong>
                  <span>{person.image_count} training images</span>
                </article>
              ))}
            </div>
          </section>
        )}

        {tab === "capture" && (
          <section className="cameraLayout">
            <div className="cameraSurface">
              <video ref={videoRef} autoPlay muted playsInline />
              <canvas ref={canvasRef} hidden />
            </div>
            <div className="panel compact">
              <div className="sectionTitle">
                <FlaskConical size={22} />
                <h2>Compression Lab</h2>
              </div>
              <button onClick={startCamera}>
                <Camera size={18} /> Start Camera
              </button>
              <button disabled={isBusy} onClick={async () => runAnalysis(await captureFrame())}>
                <FlaskConical size={18} /> Capture & Analyze
              </button>
              <label className="fileDrop">
                Or upload a test image
                <input
                  type="file"
                  accept="image/*"
                  onChange={(event) => setAnalysisFile(event.target.files?.[0] ?? null)}
                />
              </label>
              <button disabled={isBusy || !analysisFile} onClick={() => runAnalysis(analysisFile)}>
                <Upload size={18} /> Analyze Upload
              </button>
            </div>
          </section>
        )}

        {tab === "results" && (
          <section className="resultsLayout">
            <div className="summaryBand">
              <div>
                <span>Original prediction</span>
                <strong>{latest?.original_recognition.predicted_name ?? "No run yet"}</strong>
              </div>
              <div>
                <span>Recommended rank</span>
                <strong>{bestRank ?? "N/A"}</strong>
              </div>
              <div>
                <span>Confidence target</span>
                <strong>{latest ? `${Math.round(latest.accuracy_threshold * 100)}%` : "63%"}</strong>
              </div>
            </div>

            <div className="chartPanel">
              <ResponsiveContainer width="100%" height={310}>
                <LineChart data={chartRows}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="svd_rank" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="recognition.confidence" stroke="#176b87" strokeWidth={3} />
                  <Line type="monotone" dataKey="storage_reduction_percent" stroke="#c4552d" strokeWidth={3} />
                  <Line type="monotone" dataKey="psnr" stroke="#5b7f3a" strokeWidth={3} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="imageStrip">
              {chartRows.slice(0, 6).map((row) => (
                <article key={row.id} className="thumb">
                  <img src={resolveMediaUrl(row.compressed_image_url)} alt={`Rank ${row.svd_rank}`} />
                  <strong>Rank {row.svd_rank}</strong>
                  <span>{Math.round(row.recognition.confidence * 100)}% confidence</span>
                </article>
              ))}
            </div>

            <div className="tableWrap">
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                    <th>Storage Saved</th>
                    <th>PSNR</th>
                    <th>MSE</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {chartRows.map((row) => (
                    <tr key={row.id}>
                      <td>{row.svd_rank}</td>
                      <td>{row.recognition.predicted_name}</td>
                      <td>{Math.round(row.recognition.confidence * 100)}%</td>
                      <td>{row.storage_reduction_percent}%</td>
                      <td>{row.psnr.toFixed(2)}</td>
                      <td>{row.mse.toFixed(2)}</td>
                      <td>{row.processing_time_ms.toFixed(1)} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </section>
    </main>
  );
}
