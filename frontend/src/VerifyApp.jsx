import React, { useState, useRef, useCallback, useEffect } from "react";
import { Camera, Upload, X, CheckCircle2, ExternalLink, Link2, ScanFace, Search, ShieldCheck, AlertCircle } from "lucide-react";

const STAGES = [
  { key: "face", label: "Reading the face", icon: ScanFace },
  { key: "search", label: "Searching the web", icon: Search },
  { key: "chain", label: "Writing to the chain", icon: Link2 },
];

function useCamera() {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [active, setActive] = useState(false);
  const [error, setError] = useState(null);

  const start = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setActive(true);
    } catch (err) {
      setError("Camera access was denied or unavailable.");
    }
  }, []);

  const stop = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setActive(false);
  }, []);

  const capture = useCallback(() => {
    if (!videoRef.current) return null;
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);
    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.92);
    });
  }, []);

  useEffect(() => () => stop(), [stop]);

  return { videoRef, active, error, start, stop, capture };
}

// Calls the real Flask backend at /api/verify (proxied by Vite in dev).
// onStage(i) is called optimistically to animate the stepper, since the
// backend returns one final response rather than incremental progress.
async function runRealPipeline(file, onStage) {
  onStage(0);
  const stageTimer = setTimeout(() => onStage(1), 900);

  const formData = new FormData();
  formData.append("image", file);

  try {
    const res = await fetch("/api/verify", { method: "POST", body: formData });
    clearTimeout(stageTimer);
    onStage(2);

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `Request failed with status ${res.status}`);
    }
    return data;
  } catch (err) {
    clearTimeout(stageTimer);
    throw err;
  }
}

const palette = {
  cream: "#F2E8D5",
  paper: "#FBF4E4",
  coral: "#E8627E",
  coralDark: "#B33F58",
  mustard: "#F0B94A",
  green: "#2E4A3A",
  greenLight: "#4C7059",
  ink: "#1C231E",
};

function DiamondBorder({ children }) {
  return (
    <div
      style={{
        border: `2px solid ${palette.ink}`,
        borderRadius: 4,
        padding: 3,
        background: `repeating-linear-gradient(45deg, ${palette.coral} 0 8px, ${palette.mustard} 8px 16px)`,
      }}
    >
      <div style={{ background: palette.paper, borderRadius: 2 }}>{children}</div>
    </div>
  );
}

export default function VerifyApp() {
  const [tab, setTab] = useState("upload");
  const [imageData, setImageData] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [stageIndex, setStageIndex] = useState(-1);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const fileInputRef = useRef(null);
  const camera = useCamera();

  useEffect(() => {
    if (tab === "camera") camera.start();
    else camera.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab]);

  const handleFiles = (files) => {
    const file = files[0];
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      setImageData(e.target.result);
      setImageFile(file);
      setFileName(file.name);
      setStatus("idle");
      setResult(null);
      setErrorMsg(null);
    };
    reader.readAsDataURL(file);
  };

  const handleCapture = async () => {
    const blob = await camera.capture();
    if (blob) {
      const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
      const reader = new FileReader();
      reader.onload = (e) => {
        setImageData(e.target.result);
        setImageFile(file);
        setFileName(file.name);
        setStatus("idle");
        setResult(null);
        setErrorMsg(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const runPipeline = async () => {
    if (!imageFile) return;
    setStatus("running");
    setStageIndex(0);
    setErrorMsg(null);
    try {
      const data = await runRealPipeline(imageFile, setStageIndex);
      setResult(data);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err.message || "Something went wrong talking to the backend.");
      setStatus("error");
    }
  };

  const reset = () => {
    setImageData(null);
    setImageFile(null);
    setFileName(null);
    setStatus("idle");
    setStageIndex(-1);
    setResult(null);
    setErrorMsg(null);
  };

  return (
    <div
      style={{
        fontFamily: "'Archivo Narrow', 'Arial Narrow', sans-serif",
        background: palette.cream,
        minHeight: "100vh",
        padding: "32px 20px",
        display: "flex",
        justifyContent: "center",
      }}
    >
      <div style={{ width: "100%", maxWidth: 520 }}>
        <DiamondBorder>
          <div style={{ padding: "28px 28px 32px" }}>
            <header style={{ textAlign: "center", marginBottom: 24 }}>
              <div
                style={{
                  fontSize: 12,
                  letterSpacing: 3,
                  color: palette.greenLight,
                  fontWeight: 700,
                  marginBottom: 6,
                }}
              >
                HH GOA 2026 · TASK 3
              </div>
              <h1
                style={{
                  fontFamily: "'Bebas Neue', 'Archivo Black', sans-serif",
                  fontSize: 40,
                  lineHeight: 1,
                  color: palette.ink,
                  margin: 0,
                  letterSpacing: 1,
                }}
              >
                Face to Chain
              </h1>
              <p style={{ color: palette.greenLight, fontSize: 14, marginTop: 8 }}>
                Scan a face. Find the post. Lock it on-chain.
              </p>
            </header>

            <div
              style={{
                display: "flex",
                gap: 8,
                marginBottom: 18,
                background: "rgba(46,74,58,0.08)",
                padding: 4,
                borderRadius: 10,
              }}
            >
              {[
                { id: "upload", label: "Upload", icon: Upload },
                { id: "camera", label: "Camera", icon: Camera },
              ].map((t) => {
                const Icon = t.icon;
                const isActive = tab === t.id;
                return (
                  <button
                    key={t.id}
                    onClick={() => setTab(t.id)}
                    style={{
                      flex: 1,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 8,
                      padding: "10px 0",
                      border: "none",
                      borderRadius: 8,
                      fontWeight: 700,
                      fontSize: 14,
                      cursor: "pointer",
                      background: isActive ? palette.green : "transparent",
                      color: isActive ? palette.cream : palette.green,
                      transition: "background 0.15s ease",
                    }}
                  >
                    <Icon size={16} />
                    {t.label}
                  </button>
                );
              })}
            </div>

            {!imageData && tab === "upload" && (
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  handleFiles(e.dataTransfer.files);
                }}
                onClick={() => fileInputRef.current?.click()}
                style={{
                  border: `2px dashed ${dragOver ? palette.coral : palette.greenLight}`,
                  borderRadius: 10,
                  padding: "40px 20px",
                  textAlign: "center",
                  cursor: "pointer",
                  background: dragOver ? "rgba(232,98,126,0.08)" : "transparent",
                  transition: "all 0.15s ease",
                }}
              >
                <Upload size={28} color={palette.green} style={{ marginBottom: 10 }} />
                <div style={{ fontWeight: 700, color: palette.ink, fontSize: 15 }}>
                  Drop a photo here
                </div>
                <div style={{ color: palette.greenLight, fontSize: 13, marginTop: 4 }}>
                  or click to browse — jpg, png
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleFiles(e.target.files)}
                  style={{ display: "none" }}
                />
              </div>
            )}

            {!imageData && tab === "camera" && (
              <div style={{ textAlign: "center" }}>
                <div
                  style={{
                    borderRadius: 10,
                    overflow: "hidden",
                    background: palette.ink,
                    aspectRatio: "4/3",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    marginBottom: 12,
                  }}
                >
                  {camera.error ? (
                    <div style={{ color: palette.cream, fontSize: 13, padding: 20 }}>
                      {camera.error}
                    </div>
                  ) : (
                    <video
                      ref={camera.videoRef}
                      muted
                      playsInline
                      style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    />
                  )}
                </div>
                <button
                  onClick={handleCapture}
                  disabled={!camera.active}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "10px 22px",
                    borderRadius: 8,
                    border: "none",
                    background: camera.active ? palette.coral : palette.greenLight,
                    color: "#fff",
                    fontWeight: 700,
                    fontSize: 14,
                    cursor: camera.active ? "pointer" : "not-allowed",
                  }}
                >
                  <Camera size={16} />
                  Capture photo
                </button>
              </div>
            )}

            {imageData && (
              <div>
                <div style={{ position: "relative", marginBottom: 16 }}>
                  <img
                    src={imageData}
                    alt="Selected face"
                    style={{
                      width: "100%",
                      borderRadius: 10,
                      display: "block",
                      border: `2px solid ${palette.ink}`,
                    }}
                  />
                  <button
                    onClick={reset}
                    aria-label="Remove image"
                    style={{
                      position: "absolute",
                      top: 8,
                      right: 8,
                      background: palette.ink,
                      color: "#fff",
                      border: "none",
                      borderRadius: "50%",
                      width: 28,
                      height: 28,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      cursor: "pointer",
                    }}
                  >
                    <X size={16} />
                  </button>
                  <div
                    style={{
                      position: "absolute",
                      bottom: 8,
                      left: 8,
                      background: "rgba(28,35,30,0.75)",
                      color: palette.cream,
                      fontSize: 12,
                      padding: "3px 8px",
                      borderRadius: 6,
                    }}
                  >
                    {fileName}
                  </div>
                </div>

                {(status === "idle" || status === "error") && (
                  <button
                    onClick={runPipeline}
                    style={{
                      width: "100%",
                      padding: "13px 0",
                      borderRadius: 8,
                      border: "none",
                      background: palette.coral,
                      color: "#fff",
                      fontWeight: 700,
                      fontSize: 15,
                      cursor: "pointer",
                      marginBottom: 12,
                    }}
                  >
                    Run verification
                  </button>
                )}

                {status === "error" && (
                  <div
                    style={{
                      display: "flex",
                      alignItems: "flex-start",
                      gap: 8,
                      padding: "12px 14px",
                      borderRadius: 8,
                      background: "rgba(179,63,88,0.1)",
                      border: `1px solid ${palette.coralDark}`,
                      color: palette.coralDark,
                      fontSize: 13,
                      marginBottom: 4,
                    }}
                  >
                    <AlertCircle size={16} style={{ flexShrink: 0, marginTop: 1 }} />
                    <span>{errorMsg}</span>
                  </div>
                )}

                {status === "running" && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    {STAGES.map((s, i) => {
                      const Icon = s.icon;
                      const doneStage = i < stageIndex;
                      const activeStage = i === stageIndex;
                      return (
                        <div
                          key={s.key}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 10,
                            padding: "10px 12px",
                            borderRadius: 8,
                            background: activeStage ? "rgba(240,185,74,0.18)" : "transparent",
                            color: doneStage ? palette.greenLight : palette.ink,
                            fontWeight: activeStage ? 700 : 500,
                            fontSize: 14,
                          }}
                        >
                          {doneStage ? (
                            <CheckCircle2 size={18} color={palette.greenLight} />
                          ) : (
                            <Icon
                              size={18}
                              color={activeStage ? palette.coral : palette.greenLight}
                              style={activeStage ? { animation: "spin 1.4s linear infinite" } : {}}
                            />
                          )}
                          {s.label}
                        </div>
                      );
                    })}
                    <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
                  </div>
                )}

                {status === "done" && result && (
                  <div
                    style={{
                      border: `2px solid ${palette.ink}`,
                      borderRadius: 10,
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        background: palette.green,
                        color: palette.cream,
                        padding: "10px 14px",
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        fontWeight: 700,
                        fontSize: 14,
                      }}
                    >
                      <ShieldCheck size={18} color={palette.mustard} />
                      {result.chain.verified ? "Verified on-chain" : "Stored, verification pending"}
                    </div>
                    <div style={{ padding: "16px 16px 18px", background: palette.paper }}>
                      <Row label="Face encoded" value={`${result.face.encodingLength}-d vector`} />
                      <Row label="Face distance" value={result.match.faceDistance != null ? Number(result.match.faceDistance).toFixed(4) : "verified"} mono />
                      <Row label="Similarity" value={result.match.similarity != null ? `${(Number(result.match.similarity) * 100).toFixed(1)}%` : "threshold passed"} />
                      <Row label="Matching post" value={result.match.title} />
                      <a
                        href={result.match.link}
                        target="_blank"
                        rel="noreferrer"
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          gap: 5,
                          fontSize: 13,
                          color: palette.coralDark,
                          marginBottom: 12,
                          textDecoration: "none",
                        }}
                      >
                        {result.match.source}
                        <ExternalLink size={12} />
                      </a>
                      <Row label="Content hash" value={truncateMiddle(result.chain.hash)} mono />
                      <Row label="Transaction" value={truncateMiddle(result.chain.tx)} mono />
                      <Row label="Independent verification" value={result.chain.verified ? "VERIFIED = True" : "VERIFIED = False"} />
                      <Row
                        label="Recorded at"
                        value={new Date(result.chain.timestamp * 1000).toLocaleString()}
                      />
                    </div>
                  </div>
                )}

                {status === "done" && (
                  <button
                    onClick={reset}
                    style={{
                      width: "100%",
                      marginTop: 12,
                      padding: "11px 0",
                      borderRadius: 8,
                      border: `2px solid ${palette.green}`,
                      background: "transparent",
                      color: palette.green,
                      fontWeight: 700,
                      fontSize: 14,
                      cursor: "pointer",
                    }}
                  >
                    Verify another photo
                  </button>
                )}
              </div>
            )}
          </div>
        </DiamondBorder>

        <p
          style={{
            textAlign: "center",
            fontSize: 12,
            color: palette.greenLight,
            marginTop: 14,
          }}
        >
          2:47PM STUDIO · HACKER HOUSE
        </p>
      </div>
    </div>
  );
}

function Row({ label, value, mono }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        gap: 12,
        padding: "6px 0",
        borderBottom: `1px solid rgba(46,74,58,0.15)`,
        fontSize: 13,
      }}
    >
      <span style={{ color: palette.greenLight, flexShrink: 0 }}>{label}</span>
      <span
        style={{
          color: palette.ink,
          fontWeight: 600,
          textAlign: "right",
          fontFamily: mono ? "monospace" : "inherit",
          wordBreak: "break-all",
        }}
      >
        {value}
      </span>
    </div>
  );
}

function truncateMiddle(str, head = 10, tail = 8) {
  if (!str || str.length <= head + tail) return str;
  return `${str.slice(0, head)}…${str.slice(-tail)}`;
}
