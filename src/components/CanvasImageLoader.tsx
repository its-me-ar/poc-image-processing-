import React, { useRef, useState, useLayoutEffect } from "react";
import type { ChangeEvent } from "react";

import useOpenCv from "../hooks/useOpenCv";
import useBlurWorker from "../hooks/useBlurWorker";
import removeBackgroundUtil from "../utils/removeBackgroundLocal";
import { drawImageToCanvas, createScaledCanvas } from "../utils/imageUtils";
import { applyGaussianBlur as utilApplyGaussianBlur } from "../utils/imageProcessing";

// Keep file-local minimal types
declare global {
  interface Window {
    cv: any;
  }
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
  const [blurValue, setBlurValue] = useState(5); // Gaussian blur kernel
  const opencvReady = useOpenCv();
  // Local background removal state
  const [isRemovingBgLocal, setIsRemovingBgLocal] = useState(false);

  // Apply Gaussian blur safely (delegated to util)
  function applyGaussianBlur(img: HTMLImageElement, ksize: number) {
    if (!canvasRef.current || !opencvReady) return;
    return utilApplyGaussianBlur(canvasRef.current, img, ksize, {
      useWorker,
      runBlur,
      cv: window.cv,
      processingRef,
    });
  }

  // Local background removal wrapper
  async function removeBackgroundLocal() {
    if (!imgRef.current) {
      console.log("[RMBG] No image loaded, aborting.");
      return;
    }
    if (isRemovingBgLocal) {
      console.log("[RMBG] Background removal already in progress.");
      return;
    }

    setIsRemovingBgLocal(true);
    try {
      const scaled = createScaledCanvas(imgRef.current);
      await removeBackgroundUtil(scaled, (tmp) => drawImageToCanvas(canvasRef.current, tmp));
    } catch (err) {
      console.error("[RMBG] Background removal failed:", err);
      if (imgRef.current) drawImageToCanvas(canvasRef.current, imgRef.current);
    } finally {
      setIsRemovingBgLocal(false);
    }
  }

  // Update canvas whenever image or blur changes
  useLayoutEffect(() => {
    if (!imageSrc || !imgRef.current) return;
    if (opencvReady) applyGaussianBlur(imgRef.current, blurValue);
    else drawImageToCanvas(canvasRef.current, imgRef.current);
  }, [imageSrc, blurValue, opencvReady]);

  // Load image
  useLayoutEffect(() => {
    if (!imageSrc) return;
    const img = new Image();
    // Ensure CORS-safe for canvas operations
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
  }, [imageSrc, opencvReady]);

  function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    const file = files[0];
    if (!file.type.startsWith("image/")) {
      alert("Please drop an image file (png, jpeg, webp...)");
      return;
    }

    const url = URL.createObjectURL(file);
    setImageSrc(url);

    const revoke = () => URL.revokeObjectURL(url);
    const img = new Image();
    img.onload = () => setTimeout(revoke, 2000);
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
        <p className="mb-2 text-sm text-gray-700">
          Drag & drop an image, or use the picker.
        </p>

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
