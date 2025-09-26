import React, { useRef, useState, useLayoutEffect, useEffect } from "react";
import type { ChangeEvent } from "react";
import * as ort from "onnxruntime-web";

import useOpenCv from "../hooks/useOpenCv";
import useBlurWorker from "../hooks/useBlurWorker";
import removeBackgroundUtil from "../utils/removeBackgroundLocal";
import { drawImageToCanvas, createScaledCanvas } from "../utils/imageUtils";
import { applyGaussianBlur as utilApplyGaussianBlur } from "../utils/imageProcessing";

// Simple tokenizer type
interface SimpleTokenizer {
  encode: (text: string) => { ids: number[]; attentionMask: number[] };
}

// Global Canvas type
declare global {
  interface Window {
    cv: any;
  }
}

interface IntentResult {
  action: "draw_circle" | "adjust_brightness";
  color?: string;
  value?: number;
  x?: number;
  y?: number;
}

export default function CanvasImageLoader() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const processingRef = useRef<boolean>(false);

  const [useWorker, setUseWorker] = useState(false);
  const { runBlur } = useBlurWorker();

  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [blurValue, setBlurValue] = useState(5);
  const [isRemovingBgLocal, setIsRemovingBgLocal] = useState(false);
  const [command, setCommand] = useState("");

  const opencvReady = useOpenCv();

  const [onnxSession, setOnnxSession] = useState<ort.InferenceSession | null>(
    null
  );
  const [tokenizer, setTokenizer] = useState<SimpleTokenizer | null>(null);

  // Load ONNX model and tokenizer
  useEffect(() => {
    async function loadModelAndTokenizer() {
      try {
        // Ensure ONNX Runtime knows where to load .wasm from. We copy the runtime .wasm into /ort
        try {
          // prefer explicit setting if available — use absolute URL so Vite dev server serves it correctly
          if (typeof window !== "undefined" && (ort as any).env?.wasm) {
            (ort as any).env.wasm.wasmPaths = `${window.location.origin}/ort/`;
          }
        } catch (e) {
          // ignore if not supported by this ort version
        }

        const session = await ort.InferenceSession.create(
          "/onnx_model/model.onnx"
        );
        setOnnxSession(session);

        // Load tokenizer from tokenizer.json
        const tokRes = await fetch("/onnx_model/tokenizer.json");
        const tokJson = await tokRes.json();
        // tokenizer file can have vocab under model.vocab (huggingface) or top-level 'vocab'
        let vocab: Record<string, number> =
          tokJson?.model?.vocab ?? tokJson?.vocab ?? {};
        // If vocab isn't a plain object or is empty, try loading vocab.txt as a fallback
        if (!vocab || Object.keys(vocab).length === 0) {
          try {
            const vtxt = await fetch("/onnx_model/vocab.txt");
            if (vtxt.ok) {
              const txt = await vtxt.text();
              vocab = Object.create(null);
              txt.split(/\r?\n/).forEach((token, idx) => {
                if (token.length === 0) return;
                vocab[token] = idx;
              });
            }
          } catch (e) {
            // ignore and keep vocab as empty object
            vocab = vocab ?? {};
          }
        }
        const simpleTokenizer: SimpleTokenizer = {
          encode: (text: string) => {
            try {
              const cleaned = String(text || "")
                .toLowerCase()
                .trim();
              // try whitespace word split first
              const words = cleaned.length === 0 ? [] : cleaned.split(/\s+/);
              let ids: number[] = words.map((w: string) => {
                // wordpiece vocab may use subwords; fallback to unk (0) if missing
                return Number(vocab[w] ?? vocab[w.replace(/^##/, "")] ?? 0);
              });
              // if no ids found (vocab empty or words not in vocab) fallback to per-char mapping
              if (ids.length === 0 || ids.every((v) => v === 0)) {
                ids = Array.from(cleaned).map((ch) => Number(vocab[ch] ?? 0));
              }
              const attentionMask = ids.map(() => 1);
              return { ids, attentionMask };
            } catch (err) {
              // fallback: very small encoding by char codes modulo a small vocab size
              const chars = Array.from(String(text || ""));
              const ids = chars.map((c) => c.charCodeAt(0) % 100);
              return { ids, attentionMask: ids.map(() => 1) };
            }
          },
        };
        setTokenizer(simpleTokenizer);
      } catch (err) {
        console.error("Failed to load ONNX model or tokenizer", err);
      }
    }
    loadModelAndTokenizer();
  }, []);

  // Apply Gaussian blur
  function applyGaussianBlur(img: HTMLImageElement, ksize: number) {
    if (!canvasRef.current || !opencvReady) return;
    return utilApplyGaussianBlur(canvasRef.current, img, ksize, {
      useWorker,
      runBlur,
      cv: window.cv,
      processingRef,
    });
  }

  async function removeBackgroundLocal() {
    if (!imgRef.current || isRemovingBgLocal) return;
    setIsRemovingBgLocal(true);
    try {
      const scaled = createScaledCanvas(imgRef.current);
      await removeBackgroundUtil(scaled, (tmp) =>
        drawImageToCanvas(canvasRef.current, tmp)
      );
    } catch (err) {
      console.error(err);
      if (imgRef.current) drawImageToCanvas(canvasRef.current, imgRef.current);
    } finally {
      setIsRemovingBgLocal(false);
    }
  }

  useLayoutEffect(() => {
    if (!imageSrc || !imgRef.current) return;
    if (opencvReady) applyGaussianBlur(imgRef.current, blurValue);
    else drawImageToCanvas(canvasRef.current, imgRef.current);
  }, [imageSrc, blurValue, opencvReady]);

  // Create image element and draw when imageSrc changes
  useEffect(() => {
    if (!imageSrc) return;
    const img = new Image();
    img.crossOrigin = "anonymous";
    imgRef.current = img;
    img.onload = () => {
      if (opencvReady) applyGaussianBlur(img, blurValue);
      else drawImageToCanvas(canvasRef.current, img);
    };
    img.onerror = () => {
      console.error("Failed to load image");
      setImageSrc(null);
      imgRef.current = null;
    };
    img.src = imageSrc;

    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [imageSrc, opencvReady, blurValue]);

  function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    const file = files[0];
    if (!file.type.startsWith("image/")) {
      alert("Please drop an image file");
      return;
    }
    const url = URL.createObjectURL(file);
    setImageSrc(url);

    // draw a quick immediate preview (helps when the component's effect timing differs)
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      drawImageToCanvas(canvasRef.current, img);
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }

  function onFileChange(e: ChangeEvent<HTMLInputElement>) {
    handleFiles(e.target.files);
  }
  function onDragOver(e: React.DragEvent) {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    setIsDragging(true);
  }
  function onDragLeave(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
  }
  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  }

  function clearCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    setImageSrc(null);
    if (imgRef.current) {
      imgRef.current.src = "";
      imgRef.current = null;
    }
  }

  // Handle command
  async function handleCommand() {
    if (!onnxSession || !tokenizer || !canvasRef.current || !command) return;

    try {
      const { ids, attentionMask } = tokenizer.encode(command);

      // Create ONNX Tensors
      // For int64 tensors expected by your model
      const inputIdsTensor = new ort.Tensor(
        "int64",
        BigInt64Array.from(ids.map((i) => BigInt(i))),
        [1, ids.length]
      );
      const attentionMaskTensor = new ort.Tensor(
        "int64",
        BigInt64Array.from(attentionMask.map((i) => BigInt(i))),
        [1, attentionMask.length]
      );

      const results = await onnxSession.run({
        input_ids: inputIdsTensor,
        attention_mask: attentionMaskTensor,
      });

      // logits
      const logits = results.logits.data as Float32Array;
      const labels = ["adjust_brightness", "draw_circle"];
      const predictedIndex = logits.indexOf(Math.max(...logits));

      let result: IntentResult = { action: "draw_circle" };
      if (labels[predictedIndex] === "adjust_brightness")
        result = { action: "adjust_brightness", value: 20 };
      else if (labels[predictedIndex] === "draw_circle")
        result = { action: "draw_circle", color: "red", x: 100, y: 100 };

      executeIntent(result);
    } catch (err) {
      console.error("Command execution failed:", err);
    }
  }

  function executeIntent(intent: IntentResult) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    switch (intent.action) {
      case "draw_circle":
        ctx.fillStyle = intent.color || "red";
        ctx.beginPath();
        // Draw at center
        ctx.arc(centerX, centerY, 30, 0, 2 * Math.PI);
        ctx.fill();
        break;

      case "adjust_brightness":
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const factor = (intent.value || 20) / 100;
        for (let i = 0; i < imageData.data.length; i += 4) {
          imageData.data[i] = Math.min(255, imageData.data[i] * (1 + factor));
          imageData.data[i + 1] = Math.min(
            255,
            imageData.data[i + 1] * (1 + factor)
          );
          imageData.data[i + 2] = Math.min(
            255,
            imageData.data[i + 2] * (1 + factor)
          );
        }
        ctx.putImageData(imageData, 0, 0);
        break;
    }
  }

  return (
    <div
      className="fixed inset-0"
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <canvas ref={canvasRef} className="w-full h-full block bg-black" />

      <div className="absolute top-4 left-1/2 -translate-x-1/2 px-3 py-2 rounded bg-white/80 backdrop-blur text-sm font-semibold shadow">
        POC-IMAGE PROCESSING
      </div>

      <div className="absolute top-4 right-4 w-[min(420px,90vw)] rounded-xl border bg-white/90 backdrop-blur p-3 shadow-lg">
        {/* Command input */}
        <input
          type="text"
          placeholder='Type command, e.g., "draw red circle"'
          className="w-full mb-2 px-2 py-1 border rounded"
          value={command}
          onChange={(e) => setCommand(e.target.value)}
        />
        <button
          className="px-3 py-1 border rounded mb-3 w-full"
          onClick={handleCommand}
        >
          Execute Command
        </button>

        {/* Image controls */}
        <div className="flex gap-2 mb-2">
          <button
            type="button"
            className="px-3 py-2 border rounded hover:shadow"
            onClick={() => fileInputRef.current?.click()}
          >
            Choose Image
          </button>
          <button
            type="button"
            className="px-3 py-2 border rounded hover:shadow"
            onClick={clearCanvas}
          >
            Clear
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={onFileChange}
        />

        {imageSrc && (
          <>
            <div className="mt-2 text-xs text-gray-600 max-h-16 overflow-y-auto break-all">
              <span className="font-medium">Loaded:</span>{" "}
              <span className="font-mono">{imageSrc}</span>
            </div>

            <div className="mt-4 flex items-center gap-2">
              <label htmlFor="blurSlider" className="text-sm text-gray-700">
                Blur:
              </label>
              <input
                type="range"
                id="blurSlider"
                min={1}
                max={25}
                step={2}
                value={blurValue}
                onChange={(e) => setBlurValue(Number(e.target.value))}
                className="w-full"
              />
              <span className="text-sm">{blurValue}</span>
            </div>

            <div className="mt-3 flex items-center gap-2">
              <input
                id="useWorker"
                type="checkbox"
                checked={useWorker}
                onChange={(e) => setUseWorker(e.target.checked)}
              />
              <label htmlFor="useWorker" className="text-sm text-gray-700">
                Use Web Worker (Offload processing)
              </label>
            </div>

            <div className="mt-3">
              <button
                type="button"
                onClick={removeBackgroundLocal}
                disabled={isRemovingBgLocal}
                className={`px-3 py-2 rounded border ${
                  isRemovingBgLocal
                    ? "opacity-60 cursor-not-allowed"
                    : "hover:shadow"
                }`}
              >
                {isRemovingBgLocal
                  ? "Removing Background…"
                  : "Remove Background (Local)"}
              </button>
            </div>
          </>
        )}
      </div>
      {isDragging && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="px-4 py-2 bg-white/80 rounded shadow">
            Drop image to upload
          </div>
        </div>
      )}
    </div>
  );
}
