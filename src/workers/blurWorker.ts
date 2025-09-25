// OpenCV blur worker: loads opencv.js inside worker and processes ImageData off main thread
// Message protocol:
// { type: 'init' }
// { type: 'blur', id, imageData, width, height, ksize }
// Response: { type: 'ready' } | { type: 'result', id, imageData } | { type: 'error', id?, message }

let ready = false as boolean;
let loadingPromise: Promise<void> | null = null;

function postError(id: number | undefined, message: string) {
  (self as unknown as Worker).postMessage({ type: 'error', id, message });
}

function ensureOpenCV(): Promise<void> {
  if (ready) return Promise.resolve();
  if (loadingPromise) return loadingPromise;
  loadingPromise = new Promise((resolve) => {
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    importScripts('https://docs.opencv.org/4.x/opencv.js');
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    (self as any).cv = (self as any).cv || (self as any).Module;
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    (self as any).cv.onRuntimeInitialized = () => {
      ready = true;
      (self as unknown as Worker).postMessage({ type: 'ready' });
      resolve();
    };
  });
  return loadingPromise;
}

async function handleBlur(ev: MessageEvent) {
  const { id, imageData, ksize } = ev.data;
  try {
    await ensureOpenCV();
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    const cv = (self as any).cv as any;
    const src = cv.matFromImageData(imageData);
    const dst = new cv.Mat();
    const k = ksize % 2 === 0 ? ksize + 1 : ksize;
    cv.GaussianBlur(src, dst, new cv.Size(k, k), 0, 0, cv.BORDER_DEFAULT);
    const out = new ImageData(new Uint8ClampedArray(dst.data), dst.cols, dst.rows);
    src.delete();
    dst.delete();
    (self as unknown as Worker).postMessage({ type: 'result', id, imageData: out }, [out.data.buffer]);
  } catch (err: any) {
    postError(id, err?.message ?? String(err));
  }
}

(self as unknown as Worker).onmessage = async (ev: MessageEvent) => {
  const { type } = ev.data || {};
  switch (type) {
    case 'init':
      await ensureOpenCV();
      return;
    case 'blur':
      await handleBlur(ev);
      return;
    default:
      postError(undefined, 'unknown message');
  }
};


