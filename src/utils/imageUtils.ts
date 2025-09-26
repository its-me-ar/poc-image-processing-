export function drawImageToCanvas(
  canvas: HTMLCanvasElement | null,
  img: HTMLImageElement | HTMLCanvasElement
) {
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
    (img as HTMLImageElement).naturalWidth ?? (img as HTMLCanvasElement).width;
  const ih =
    (img as HTMLImageElement).naturalHeight ?? (img as HTMLCanvasElement).height;

  const scale = Math.min(cw / iw, ch / ih);
  const renderedWidth = Math.round(iw * scale);
  const renderedHeight = Math.round(ih * scale);
  const dx = Math.round((cw - renderedWidth) / 2);
  const dy = Math.round((ch - renderedHeight) / 2);

  ctx.drawImage(img, 0, 0, iw, ih, dx, dy, renderedWidth, renderedHeight);
}

export function createScaledCanvas(img: HTMLImageElement): HTMLCanvasElement {
  const MAX_DIMENSION = 1600; // cap longest side
  const MAX_PIXELS = 3_000_000; // ~3MP

  const iw = img.naturalWidth;
  const ih = img.naturalHeight;

  let targetW = iw;
  let targetH = ih;

  const dimScale = Math.min(1, MAX_DIMENSION / Math.max(iw, ih));
  targetW = Math.round(iw * dimScale);
  targetH = Math.round(ih * dimScale);

  const pixels = targetW * targetH;
  if (pixels > MAX_PIXELS) {
    const pxScale = Math.sqrt(MAX_PIXELS / pixels);
    targetW = Math.max(1, Math.floor(targetW * pxScale));
    targetH = Math.max(1, Math.floor(targetH * pxScale));
  }

  if (targetW === iw && targetH === ih) {
    const c0 = document.createElement("canvas");
    c0.width = iw;
    c0.height = ih;
    const c0ctx = c0.getContext("2d");
    if (c0ctx) c0ctx.drawImage(img, 0, 0);
    return c0;
  }

  const canvas = document.createElement("canvas");
  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext("2d", { alpha: true });
  if (ctx) {
    ctx.imageSmoothingEnabled = true;
    // @ts-ignore imageSmoothingQuality is well-known but some lib defs omit it
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, iw, ih, 0, 0, targetW, targetH);
  }
  return canvas;
}
