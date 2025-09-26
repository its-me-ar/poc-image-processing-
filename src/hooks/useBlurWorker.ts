import { useRef } from "react";

export default function useBlurWorker() {
  const workerRef = useRef<Worker | null>(null);
  const jobIdRef = useRef(0);

  function ensureWorker() {
    if (workerRef.current) return;
    const workerUrl = new URL("../workers/blurWorker.ts", import.meta.url);
    workerRef.current = new Worker(workerUrl, { type: "classic" } as any);
    workerRef.current.postMessage({ type: "init" });
  }

  function runBlur(imageData: ImageData, width: number, height: number, ksize: number) {
    ensureWorker();
    const id = ++jobIdRef.current;

    return new Promise<ImageData>((resolve, reject) => {
      if (!workerRef.current) return reject(new Error("Worker not available"));
      const handle = (ev: MessageEvent) => {
        const { type, id: retId, imageData: out, message } = ev.data || {};
        if (type === "result" && retId === id) {
          resolve(out);
          workerRef.current?.removeEventListener("message", handle);
        } else if (type === "error") {
          reject(new Error(message || "worker error"));
          workerRef.current?.removeEventListener("message", handle);
        }
      };

      workerRef.current.addEventListener("message", handle);
      workerRef.current.postMessage({ type: "blur", id, imageData, width, height, ksize });
    });
  }

  return { runBlur };
}
