export type RemoveBgOptions = {
  // allow advanced configuration if needed later
};

export default async function removeBackgroundLocal(
  canvasOrImage: HTMLCanvasElement | HTMLImageElement,
  drawCallback: (c: HTMLCanvasElement) => void
) {
  console.log("[RMBG] Starting background removal (util)...");

  const scaledCanvas = ((): HTMLCanvasElement => {
    if (canvasOrImage instanceof HTMLCanvasElement) return canvasOrImage;
    // simple scaling logic: draw into a canvas with same size
    const c = document.createElement("canvas");
    c.width = (canvasOrImage as HTMLImageElement).naturalWidth;
    c.height = (canvasOrImage as HTMLImageElement).naturalHeight;
    const ctx = c.getContext("2d");
    if (ctx) ctx.drawImage(canvasOrImage, 0, 0);
    return c;
  })();

  try {
    const { AutoModel, AutoProcessor, RawImage, env } = await import(
      "@huggingface/transformers"
    );

    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    // @ts-ignore
    env.backends.onnx.wasm.proxy = true;

    const model = await AutoModel.from_pretrained("briaai/RMBG-1.4", {
      // @ts-ignore
      config: { model_type: "custom" },
    });

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

    const image = await RawImage.fromCanvas(scaledCanvas as any);
    const { pixel_values } = await processor(image as any);
    const { output } = await model({ input: pixel_values });

    const mask = await RawImage.fromTensor(
      output[0].mul(255).to("uint8")
    ).resize(scaledCanvas.width, scaledCanvas.height);

    image.putAlpha(mask as any);

    const tmp = document.createElement("canvas");
    tmp.width = scaledCanvas.width;
    tmp.height = scaledCanvas.height;
    const tctx = tmp.getContext("2d");
    if (!tctx) throw new Error("Failed to create canvas context");
    tctx.drawImage(image.toCanvas(), 0, 0);

    drawCallback(tmp);
    console.log("[RMBG] Background removal finished (util)");
  } catch (err) {
    console.error("[RMBG] removeBackgroundLocal failed:", err);
    throw err;
  }
}
