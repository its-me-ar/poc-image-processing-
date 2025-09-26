import { createScaledCanvas, drawImageToCanvas } from "./imageUtils";

// Lightweight typing to avoid pulling in DOM lib extras
type CanvasRef = HTMLCanvasElement | null;

export type ApplyBlurOptions = {
  useWorker: boolean;
  runBlur?: (
    imageData: ImageData,
    width: number,
    height: number,
    ksize: number
  ) => Promise<ImageData>;
  cv?: any; // OpenCV global
  processingRef?: { current: boolean };
};

export async function applyGaussianBlur(
  canvas: CanvasRef,
  img: HTMLImageElement,
  ksize: number,
  options: ApplyBlurOptions
) {
  const { useWorker, runBlur, cv, processingRef } = options;
  if (!canvas || !cv) return;
  if (processingRef && processingRef.current) return;
  if (processingRef) processingRef.current = true;

  try {
    if (!img.naturalWidth || !img.naturalHeight) return;

    const scaled = createScaledCanvas(img);

    if (useWorker && runBlur) {
      const w = scaled.width;
      const h = scaled.height;
      const ctx = scaled.getContext("2d");
      const imageData = ctx?.getImageData(0, 0, w, h);
      if (!imageData) return;

      try {
        const out = await runBlur(imageData, w, h, ksize);
        const tmp = document.createElement("canvas");
        tmp.width = w;
        tmp.height = h;
        const tctx = tmp.getContext("2d");
        if (tctx) tctx.putImageData(out, 0, 0);
        drawImageToCanvas(canvas, tmp);
      } catch (err) {
        console.error("Worker error:", err);
        drawImageToCanvas(canvas, scaled);
      } finally {
        if (processingRef) processingRef.current = false;
      }

      return;
    }

    const src = cv.imread(scaled);
    const dst = new cv.Mat();

    const k = ksize % 2 === 0 ? ksize + 1 : ksize;
    cv.GaussianBlur(src, dst, new cv.Size(k, k), 0, 0, cv.BORDER_DEFAULT);

    const tmp = document.createElement("canvas");
    tmp.width = dst.cols;
    tmp.height = dst.rows;
    cv.imshow(tmp, dst);

    drawImageToCanvas(canvas, tmp);

    src.delete();
    dst.delete();
  } catch (err) {
    console.error("OpenCV error:", err);
    drawImageToCanvas(canvas, img);
  } finally {
    if (processingRef) processingRef.current = false;
  }
}
