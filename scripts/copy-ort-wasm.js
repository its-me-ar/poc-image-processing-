const fs = require('fs');
const path = require('path');

// Copies ONNX Runtime WASM artifacts from node_modules to public/ort
try {
  const pkgDir = path.resolve(__dirname, '..');
  const srcDir = path.join(pkgDir, 'node_modules', 'onnxruntime-web', 'dist');
  const destDir = path.join(pkgDir, 'public', 'ort');
  if (!fs.existsSync(srcDir)) {
    console.warn('onnxruntime-web not found in node_modules; skipping wasm copy.');
    process.exit(0);
  }
  if (!fs.existsSync(destDir)) fs.mkdirSync(destDir, { recursive: true });

  // copy wasm plus runtime helper modules (.mjs/.js) that onnxruntime dynamically imports
  const files = fs.readdirSync(srcDir).filter((f) => f.endsWith('.wasm') || f.endsWith('.mjs') || f.endsWith('.js'));
  files.forEach((f) => {
    const src = path.join(srcDir, f);
    const dest = path.join(destDir, f);
    try {
      fs.copyFileSync(src, dest);
      console.log('Copied', f);
    } catch (err) {
      console.warn('Failed to copy', f, err.message);
    }
  });
} catch (err) {
  console.error('Error while copying ONNX WASM files:', err);
}
