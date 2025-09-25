import React, { useRef, useState, useLayoutEffect, useEffect } from "react";
import type { ChangeEvent } from "react";

// Declare global cv for TypeScript
declare global {
  interface Window {
    cv: any;
    __opencvReadyPromise?: Promise<void>;
    __opencvScriptLoading?: boolean;
  }
}

export default function CanvasImageLoader() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const processingRef = useRef<boolean>(false);
  const [useWorker, setUseWorker] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  const jobIdRef = useRef<number>(0);

  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [blurValue, setBlurValue] = useState(5); // Gaussian blur kernel
  const [opencvReady, setOpenCvReady] = useState(false);
  // Local background removal state
  const [isRemovingBgLocal, setIsRemovingBgLocal] = useState(false);

  // Load OpenCV.js only once and wait for WASM runtime initialization
  useEffect(() => {
    const markReady = () => setOpenCvReady(true);

    // If runtime already initialized
    try {
      if (
        window.cv &&
        typeof (window.cv as any).getBuildInformation === "function"
      ) {
        markReady();
        return;
      }
    } catch {}

    // If there is an existing shared promise, reuse it
    if (window.__opencvReadyPromise) {
      window.__opencvReadyPromise.then(markReady);
      return;
    }

    // Create a shared promise to avoid double init
    let resolveReady: () => void;
    window.__opencvReadyPromise = new Promise<void>((resolve) => {
      resolveReady = resolve;
    });
    window.__opencvReadyPromise.then(markReady);

    // Hook runtime init callback once script finishes loading
    const onScriptLoad = () => {
      if (!window.cv) return;
      (window.cv as any).onRuntimeInitialized = () => {
        resolveReady();
      };
    };

    // Only append script once
    if (!window.__opencvScriptLoading) {
      window.__opencvScriptLoading = true;
      const existing = document.querySelector('script[src*="opencv.js"]');
      if (existing) {
        // If script tag exists from HMR, just attach
        if ((existing as HTMLScriptElement).dataset._opencvHooked !== "1") {
          existing.addEventListener("load", onScriptLoad, { once: true });
          (existing as HTMLScriptElement).dataset._opencvHooked = "1";
        }
      } else {
        const script = document.createElement("script");
        script.src = "https://docs.opencv.org/4.x/opencv.js";
        script.async = true;
        script.dataset._opencvHooked = "1";
        script.addEventListener("load", onScriptLoad, { once: true });
        document.body.appendChild(script);
      }
    } else {
      // Script is loading elsewhere; we rely on the shared promise
    }
    // Do NOT remove script on unmount to avoid IntVector error
  }, []);

  // Draw image/canvas to main canvas responsively
  function drawImageToCanvas(img: HTMLImageElement | HTMLCanvasElement) {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const cw = rect.width;
    const ch = rect.height;
    const iw =
      (img as HTMLImageElement).naturalWidth ??
      (img as HTMLCanvasElement).width;
    const ih =
      (img as HTMLImageElement).naturalHeight ??
      (img as HTMLCanvasElement).height;

    const scale = Math.min(cw / iw, ch / ih);
    const renderedWidth = Math.round(iw * scale);
    const renderedHeight = Math.round(ih * scale);
    const dx = Math.round((cw - renderedWidth) / 2);
    const dy = Math.round((ch - renderedHeight) / 2);

    ctx.drawImage(img, 0, 0, iw, ih, dx, dy, renderedWidth, renderedHeight);
  }

  // Apply Gaussian blur safely
  function applyGaussianBlur(img: HTMLImageElement, ksize: number) {
    if (!canvasRef.current || !opencvReady) return;
    if (processingRef.current) return;
    processingRef.current = true;

    try {
      if (!img.naturalWidth || !img.naturalHeight) return;

      // Downscale very large images to prevent memory spikes and UI hangs
      const scaled = createScaledCanvas(img);

      if (useWorker) {
        ensureWorker();
        const w = scaled.width;
        const h = scaled.height;
        const ctx = scaled.getContext("2d");
        const imageData = ctx?.getImageData(0, 0, w, h);
        const id = ++jobIdRef.current;
        if (!imageData || !workerRef.current) return;
        workerRef.current.onmessage = (ev: MessageEvent) => {
          const { type, id: retId, imageData: out, message } = ev.data || {};
          if (type === "result" && retId === id) {
            // Draw to temp canvas
            const tmp = document.createElement("canvas");
            tmp.width = w;
            tmp.height = h;
            const tctx = tmp.getContext("2d");
            if (tctx) tctx.putImageData(out, 0, 0);
            drawImageToCanvas(tmp);
            processingRef.current = false;
          } else if (type === "error") {
            console.error("Worker error:", message);
            drawImageToCanvas(scaled);
            processingRef.current = false;
          }
        };
        workerRef.current.postMessage({
          type: "blur",
          id,
          imageData,
          width: w,
          height: h,
          ksize,
        });
        return; // early; will unset processing flag in handler
      }

      const src = cv.imread(scaled);
      const dst = new cv.Mat();

      const k = ksize % 2 === 0 ? ksize + 1 : ksize;
      cv.GaussianBlur(src, dst, new cv.Size(k, k), 0, 0, cv.BORDER_DEFAULT);

      // Render result to an intermediate canvas (pixel-perfect)
      const tmp = document.createElement("canvas");
      tmp.width = dst.cols;
      tmp.height = dst.rows;
      cv.imshow(tmp, dst);

      // Then draw responsively to main canvas
      drawImageToCanvas(tmp);

      src.delete();
      dst.delete();
    } catch (err) {
      console.error("OpenCV error:", err);
      drawImageToCanvas(img); // fallback
    } finally {
      processingRef.current = false;
    }
  }

  // Create a scaled canvas to cap size and memory consumption
  function createScaledCanvas(img: HTMLImageElement): HTMLCanvasElement {
    const MAX_DIMENSION = 1600; // cap longest side
    const MAX_PIXELS = 3_000_000; // ~3MP

    const iw = img.naturalWidth;
    const ih = img.naturalHeight;

    let targetW = iw;
    let targetH = ih;

    // First cap by max dimension
    const dimScale = Math.min(1, MAX_DIMENSION / Math.max(iw, ih));
    targetW = Math.round(iw * dimScale);
    targetH = Math.round(ih * dimScale);

    // Then cap by total pixels
    const pixels = targetW * targetH;
    if (pixels > MAX_PIXELS) {
      const pxScale = Math.sqrt(MAX_PIXELS / pixels);
      targetW = Math.max(1, Math.floor(targetW * pxScale));
      targetH = Math.max(1, Math.floor(targetH * pxScale));
    }

    if (targetW === iw && targetH === ih) {
      // No scaling needed; draw once onto a canvas to ensure consistent input
      const c0 = document.createElement("canvas");
      c0.width = iw;
      c0.height = ih;
      const c0ctx = c0.getContext("2d");
      if (c0ctx) c0ctx.drawImage(img, 0, 0);
      return c0;
    }

    // Use a canvas to scale down
    const canvas = document.createElement("canvas");
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (ctx) {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(img, 0, 0, iw, ih, 0, 0, targetW, targetH);
    }
    return canvas;
  }

  // Lazy-create worker
  function ensureWorker() {
    if (workerRef.current) return;
    const workerUrl = new URL("../workers/blurWorker.ts", import.meta.url);
    workerRef.current = new Worker(workerUrl, {
      type: "classic" as WorkerType,
    } as any);
    // Kick off init (no-op if already ready)
    workerRef.current.postMessage({ type: "init" });
  }

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
    console.log("[RMBG] Starting background removal...");

    try {
      const scaled = createScaledCanvas(imgRef.current);
      console.log("[RMBG] Image scaled:", scaled.width, "x", scaled.height);

      const { AutoModel, AutoProcessor, RawImage, env } = await import(
        "@huggingface/transformers"
      );

      console.log("[RMBG] Configuring transformers.js environment...");
      env.allowLocalModels = false; // skip local ONNX
      env.allowRemoteModels = true; // use Hugging Face Hub
      // @ts-ignore
      env.backends.onnx.wasm.proxy = true;

      console.log("[RMBG] Loading model from Hugging Face Hub...");
      const model = await AutoModel.from_pretrained("briaai/RMBG-1.4", {
        // @ts-ignore
        config: { model_type: "custom" },
      });
      console.log("[RMBG] Model loaded.");

      console.log("[RMBG] Loading processor...");
      const processor = await AutoProcessor.from_pretrained("briaai/RMBG-1.4", {
        config: {
          do_normalize: true,
          do_pad: false,
          do_rescale: true,
          do_resize: true,
          image_mean: [0.5, 0.5, 0.5],
          feature_extractor_type: "ImageFeatureExtractor",
          image_std: [1, 1, 1],
          resample: 2,
          rescale_factor: 0.00392156862745098,
          size: { width: 1024, height: 1024 },
        },
      });
      console.log("[RMBG] Processor loaded.");

      console.log("[RMBG] Converting image to RawImage...");
      const image = await RawImage.fromCanvas(scaled);

      console.log("[RMBG] Preprocessing image...");
      const { pixel_values } = await processor(image);

      console.log("[RMBG] Running model inference...");
      const { output } = await model({ input: pixel_values });
      console.log("[RMBG] Model inference completed.");

      console.log("[RMBG] Resizing mask to original canvas size...");
      const mask = await RawImage.fromTensor(
        output[0].mul(255).to("uint8")
      ).resize(scaled.width, scaled.height);

      console.log("[RMBG] Applying alpha mask to image...");
      image.putAlpha(mask);

      console.log("[RMBG] Rendering result to canvas...");
      const tmp = document.createElement("canvas");
      tmp.width = scaled.width;
      tmp.height = scaled.height;
      const tctx = tmp.getContext("2d");
      if (!tctx) {
        console.log("[RMBG] Failed to get canvas context.");
        return;
      }
      tctx.drawImage(image.toCanvas(), 0, 0);

      drawImageToCanvas(tmp);
      console.log("[RMBG] Background removal done!");
    } catch (err) {
      console.error("[RMBG] Background removal failed:", err);
      if (imgRef.current) drawImageToCanvas(imgRef.current);
    } finally {
      setIsRemovingBgLocal(false);
      console.log("[RMBG] Background removal flag reset.");
    }
  }

  // Update canvas whenever image or blur changes
  useLayoutEffect(() => {
    if (!imageSrc || !imgRef.current) return;
    if (opencvReady) applyGaussianBlur(imgRef.current, blurValue);
    else drawImageToCanvas(imgRef.current);
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
      else drawImageToCanvas(img);
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
