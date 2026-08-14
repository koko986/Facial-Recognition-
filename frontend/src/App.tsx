import { useEffect, useMemo, useRef, useState } from "react";
import { BarChart3, Camera, CircleUserRound, FlaskConical, RefreshCcw, Search, Trash2, Upload, X } from "lucide-react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { analyzeImage, deletePerson, fetchExperiments, fetchPeople, recognizeImage, registerPerson, resolveMediaUrl } from "./api";
import type { AnalyzeResponse, ExperimentResult, Person, RecognitionResult } from "./types";

type Tab = "register" | "capture" | "results";

interface LiveRecognition {
  recognition: RecognitionResult;
  updatedAt: number;
  faceBox: { x: number; y: number; width: number; height: number } | null;
  frameWidth: number;
  frameHeight: number;
}

const AUTO_RECOGNITION_INTERVAL_MS = 1200;

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
  const [cameraOn, setCameraOn] = useState(false);
  const [autoRecognizing, setAutoRecognizing] = useState(false);
  const [liveRecognition, setLiveRecognition] = useState<LiveRecognition | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recognitionTimerRef = useRef<number | null>(null);
  const cameraOnRef = useRef(false);
  const autoRecognizingRef = useRef(false);

  const refresh = async () => {
    const [peopleRows, experimentRows] = await Promise.all([fetchPeople(), fetchExperiments()]);
    setPeople(peopleRows);
    setExperiments(experimentRows.slice(-100));
  };

  useEffect(() => {
    refresh().catch((error) => setStatus(error.message));
    return () => stopCamera();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const captureFrame = async (): Promise<File | null> => {
    if (!videoRef.current || !canvasRef.current) return null;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 960;
    canvas.height = video.videoHeight || 720;
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    return blob ? new File([blob], "webcam-capture.jpg", { type: "image/jpeg" }) : null;
  };

  const stopCamera = () => {
    if (recognitionTimerRef.current !== null) {
      window.clearInterval(recognitionTimerRef.current);
      recognitionTimerRef.current = null;
    }
    autoRecognizingRef.current = false;
    cameraOnRef.current = false;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
    setAutoRecognizing(false);
    setLiveRecognition(null);
    clearOverlay();
  };

  const clearOverlay = () => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const ctx = overlay.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, overlay.width, overlay.height);
  };

  const drawOverlay = () => {
    const overlay = overlayRef.current;
    const video = videoRef.current;
    if (!overlay || !video) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;

    const displayW = video.clientWidth;
    const displayH = video.clientHeight;
    if (displayW === 0 || displayH === 0) return;
    overlay.width = displayW;
    overlay.height = displayH;
    ctx.clearRect(0, 0, displayW, displayH);

    if (!liveRecognition || !liveRecognition.faceBox) return;

    const box = liveRecognition.faceBox;
    const frameW = liveRecognition.frameWidth;
    const frameH = liveRecognition.frameHeight;
    if (!frameW || !frameH) return;

    // Map the face box from the captured frame coordinates to the displayed video size.
    const scaleX = displayW / frameW;
    const scaleY = displayH / frameH;
    const x = box.x * scaleX;
    const y = box.y * scaleY;
    const w = box.width * scaleX;
    const h = box.height * scaleY;

    const accepted = liveRecognition.recognition.accepted;
    const color = accepted ? "#2ee6a8" : "#ff5d5d";
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    // Corner accents.
    const corner = 18;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(x, y + corner); ctx.lineTo(x, y); ctx.lineTo(x + corner, y);
    ctx.moveTo(x + w - corner, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + corner);
    ctx.moveTo(x + w, y + h - corner); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - corner, y + h);
    ctx.moveTo(x + corner, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - corner);
    ctx.stroke();

    // Name label above the box.
    const label = `${liveRecognition.recognition.predicted_name}  ${Math.round(liveRecognition.recognition.confidence * 100)}%`;
    ctx.font = "700 15px Inter, sans-serif";
    const textW = ctx.measureText(label).width + 20;
    const labelH = 28;
    const labelX = x;
    const labelY = Math.max(0, y - labelH - 8);
    ctx.fillStyle = "rgba(10, 18, 16, 0.85)";
    ctx.fillRect(labelX, labelY, textW, labelH);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(labelX, labelY, textW, labelH);
    ctx.fillStyle = color;
    ctx.textBaseline = "middle";
    ctx.fillText(label, labelX + 10, labelY + labelH / 2);
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 720 }, audio: false });
      streamRef.current = stream;
      cameraOnRef.current = true;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraOn(true);
      setStatus("Camera on. Start live recognition to identify people.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not start camera.");
    }
  };

  const runAutoRecognitionTick = async () => {
    if (!cameraOnRef.current || !autoRecognizingRef.current) return;
    const file = await captureFrame();
    if (!file) return;
    try {
      const response = await recognizeImage(file);
      const video = videoRef.current;
      setLiveRecognition({
        recognition: response.recognition,
        updatedAt: Date.now(),
        faceBox: response.face_box,
        frameWidth: response.frame_width ?? video?.videoWidth ?? 960,
        frameHeight: response.frame_height ?? video?.videoHeight ?? 720,
      });
    } catch {
      // Silent — keep the previous live result; the camera keeps trying.
    }
  };

  const toggleAutoRecognition = () => {
    if (autoRecognizingRef.current) {
      if (recognitionTimerRef.current !== null) {
        window.clearInterval(recognitionTimerRef.current);
        recognitionTimerRef.current = null;
      }
      autoRecognizingRef.current = false;
      setAutoRecognizing(false);
      setLiveRecognition(null);
      clearOverlay();
      setStatus("Live recognition stopped.");
    } else {
      if (!cameraOnRef.current) {
        setStatus("Start the camera first, then enable live recognition.");
        return;
      }
      autoRecognizingRef.current = true;
      setAutoRecognizing(true);
      setStatus("Live recognition running — looking for registered faces...");
      runAutoRecognitionTick();
      recognitionTimerRef.current = window.setInterval(runAutoRecognitionTick, AUTO_RECOGNITION_INTERVAL_MS);
    }
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

  const handleDeletePerson = async (person: Person) => {
    if (!window.confirm(`Delete ${person.name} and all their training images?`)) return;
    setIsBusy(true);
    try {
      await deletePerson(person.id);
      await refresh();
      setStatus(`${person.name} deleted.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Delete failed.");
    } finally {
      setIsBusy(false);
    }
  };

  const chartRows = useMemo(() => latest?.results ?? experiments.slice(-12), [latest, experiments]);
  const acceptedRows = chartRows.filter((row) => row.recognition.accepted);
  const bestRank = latest?.recommended_rank ?? acceptedRows.sort((a, b) => a.svd_rank - b.svd_rank)[0]?.svd_rank ?? null;

  useEffect(() => {
    drawOverlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveRecognition, cameraOn]);

  return (
    <main>
      <aside className="sidebar">
        <div>
          <p className="eyebrow">Research Demo</p>
          <h1>SVD FaceVault</h1>
        </div>
        <nav>
          <button className={tab === "register" ? "active" : ""} onClick={() => { setTab("register"); stopCamera(); }} title="Register">
            <CircleUserRound size={20} /> Register
          </button>
          <button className={tab === "capture" ? "active" : ""} onClick={() => setTab("capture")} title="Capture">
            <Camera size={20} /> Capture
          </button>
          <button className={tab === "results" ? "active" : ""} onClick={() => { setTab("results"); stopCamera(); }} title="Results">
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
                  <button className="deleteButton" disabled={isBusy} onClick={() => handleDeletePerson(person)} title={`Delete ${person.name}`}>
                    <Trash2 size={14} /> Delete
                  </button>
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
              <canvas ref={overlayRef} className="faceOverlay" />

              {!cameraOn && (
                <div className="cameraPlaceholder">
                  <Camera size={40} />
                  <p>Camera is off. Press Start Camera.</p>
                </div>
              )}
            </div>

            <div className="panel compact">
              <div className="sectionTitle">
                <FlaskConical size={22} />
                <h2>Compression Lab</h2>
              </div>

              {!cameraOn ? (
                <button onClick={startCamera}>
                  <Camera size={18} /> Start Camera
                </button>
              ) : (
                <button className="outlineButton" onClick={stopCamera}>
                  <X size={18} /> Stop Camera
                </button>
              )}

              <button
                className={autoRecognizing ? "activeButton" : ""}
                disabled={!cameraOn}
                onClick={toggleAutoRecognition}
              >
                <Search size={18} /> {autoRecognizing ? "Stop Live Recognition" : "Start Live Recognition"}
              </button>

              <button disabled={isBusy || !cameraOn} onClick={async () => runAnalysis(await captureFrame())}>
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

              {liveRecognition && (
                <div className="liveResult">
                  <strong>Last live match</strong>
                  <span className={liveRecognition.recognition.accepted ? "accept" : "reject"}>
                    {liveRecognition.recognition.predicted_name} — {Math.round(liveRecognition.recognition.confidence * 100)}%
                  </span>
                  <span className="methodTag">{liveRecognition.recognition.method} · {liveRecognition.recognition.accepted ? "accepted" : "rejected"}</span>
                </div>
              )}
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