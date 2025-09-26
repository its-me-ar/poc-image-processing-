import { useEffect, useState } from "react";

declare global {
  interface Window {
    cv: any;
    __opencvReadyPromise?: Promise<void>;
    __opencvScriptLoading?: boolean;
  }
}

export default function useOpenCv() {
  const [opencvReady, setOpenCvReady] = useState(false);

  useEffect(() => {
    const markReady = () => setOpenCvReady(true);

    try {
      if (window.cv && typeof (window.cv as any).getBuildInformation === "function") {
        markReady();
        return;
      }
    } catch {}

    if (window.__opencvReadyPromise) {
      window.__opencvReadyPromise.then(markReady);
      return;
    }

    let resolveReady: () => void;
    window.__opencvReadyPromise = new Promise<void>((resolve) => {
      resolveReady = resolve;
    });
    window.__opencvReadyPromise.then(markReady);

    const onScriptLoad = () => {
      if (!window.cv) return;
      (window.cv as any).onRuntimeInitialized = () => {
        resolveReady();
      };
    };

    if (!window.__opencvScriptLoading) {
      window.__opencvScriptLoading = true;
      const existing = document.querySelector('script[src*="opencv.js"]');
      if (existing) {
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
    }
  }, []);

  return opencvReady;
}
