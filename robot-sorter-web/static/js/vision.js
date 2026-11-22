// Cámaras que nunca se usan (webcam integrada del portátil).
const EXCLUDED_CAMERAS = /HP Wide Vision|Integrated|Built-in/i;
const ANALYSIS_W = 160;
const ANALYSIS_H = 120;

// Una cámara de la estación. Tres estados:
//   live    -> vídeo del navegador (y, en la frontal, detección sobre la mesa calibrada)
//   model   -> la cámara la usa el modelo; se muestran las imágenes que ve él
//   offline -> sin cámara
class CameraView {
  constructor(root, preferredCamera, { detect = false } = {}) {
    this.root = root;
    this.preferredCamera = preferredCamera;
    this.detectEnabled = detect;
    this.video = root.querySelector("video");
    this.canvas = root.querySelector("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.preview = root.querySelector("img.preview");
    this.badge = root.querySelector(".badge");
    this.detectionLabel = root.querySelector(".detection");
    this.mode = "offline";
    this.stream = null;
    this.requestId = 0;
    this.threshold = 30;
    this.minPixelArea = 3500;
    this.background = null;
    this.box = null;
    this.overlay = null;       // imagen de referencia superpuesta (alinear la cámara)
    this.overlayAlpha = 0.5;
    this.onStatus = () => {};

    this.analysis = document.createElement("canvas");
    this.analysis.width = ANALYSIS_W;
    this.analysis.height = ANALYSIS_H;
    this.analysisCtx = this.analysis.getContext("2d", { willReadFrequently: true });

    // 15 imágenes por segundo bastan para ver y analizar la mesa.
    let last = 0;
    const loop = now => {
      this.animation = requestAnimationFrame(loop);
      if (document.hidden || now - last < 1000 / 15) return;
      last = now;
      if (this.mode === "live" && this.video.readyState >= 2) this.drawFrame();
    };
    this.animation = requestAnimationFrame(loop);
    window.addEventListener("pagehide", () => {
      this.requestId++;
      this.stopStream();
      cancelAnimationFrame(this.animation);
    });
  }

  setBadge(text, kind = "") {
    this.badge.textContent = text;
    this.badge.className = `badge ${kind}`;
  }

  stopStream() {
    if (this.stream) this.stream.getTracks().forEach(track => track.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  // Elige la cámara por nombre sin abrir antes la del portátil.
  async pickCamera() {
    const list = async () => (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput");
    let devices = await list();
    // Sin permiso, el navegador oculta nombres: se pide permiso un instante y se vuelve a listar.
    if (!devices.some(d => d.label && d.deviceId)) {
      const probe = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      probe.getTracks().forEach(track => track.stop());
      devices = await list();
    }
    devices = devices.filter(d => d.deviceId && !EXCLUDED_CAMERAS.test(d.label));
    const chosen = devices.find(d => this.preferredCamera.test(d.label));
    if (!chosen) {
      const seen = devices.map(d => d.label || "sin nombre").join(", ") || "ninguna";
      throw new Error(`No se encuentra la cámara «${this.preferredCamera.source}». Hay: ${seen}.`);
    }
    return chosen.deviceId;
  }

  async connect() {
    const requestId = ++this.requestId;
    this.stopStream();
    this.showPreview(false);
    this.setBadge("Conectando…");
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Abre la web en localhost para usar las cámaras.");
      const deviceId = await this.pickCamera();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
      });
      // Si mientras tanto se ha pedido otra cosa (p. ej. soltar la cámara), se descarta.
      if (requestId !== this.requestId) {
        stream.getTracks().forEach(track => track.stop());
        return false;
      }
      this.stream = stream;
      this.video.srcObject = stream;
      await this.video.play();
      if (requestId !== this.requestId) return false;
      this.mode = "live";
      this.error = null;
      this.setBadge("En directo", "live");
      this.onStatus();
      return true;
    } catch (error) {
      if (requestId !== this.requestId) return false;
      this.stopStream();
      this.mode = "offline";
      const messages = { NotAllowedError: "Permiso de cámara denegado.", NotReadableError: "Cámara ocupada por otro programa." };
      this.error = messages[error.name] || error.message;
      this.setBadge("Sin cámara");
      this.onStatus();
      return false;
    }
  }

  // Suelta la cámara para que la use el modelo y muestra lo que ve él.
  release() {
    this.requestId++;
    this.stopStream();
    this.mode = "model";
    this.box = null;
    if (this.detectionLabel) this.detectionLabel.hidden = true;
    this.setBadge("Vista del modelo", "model");
    this.onStatus();
  }

  showPreview(show) {
    this.preview.hidden = !show;
    this.canvas.hidden = show;
  }

  updatePreview(url) {
    if (this.mode !== "model") return;
    this.showPreview(true);
    this.preview.src = url;
  }

  drawFrame() {
    const { videoWidth: w, videoHeight: h } = this.video;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.ctx.drawImage(this.video, 0, 0, w, h);
    if (this.overlay) {
      this.ctx.globalAlpha = this.overlayAlpha;
      this.ctx.drawImage(this.overlay, 0, 0, w, h);
      this.ctx.globalAlpha = 1;
    }
    if (!this.detectEnabled || !this.background) return;
    const box = this.detect();
    if (box) {
      // Suavizado ligero para que el recuadro no tiemble.
      if (this.box) for (const key of ["x", "y", "w", "h"]) this.box[key] += (box[key] - this.box[key]) * .3;
      else this.box = { ...box };
      this.ctx.strokeStyle = "#f4a259";
      this.ctx.lineWidth = Math.max(3, w / 320);
      this.ctx.strokeRect(this.box.x, this.box.y, this.box.w, this.box.h);
    } else this.box = null;
    if (this.detectionLabel) this.detectionLabel.hidden = !this.box;
  }

  // Fotograma actual de la cámara (sin superposiciones) en JPEG.
  snapshot() {
    if (this.mode !== "live" || this.video.readyState < 2) return Promise.resolve(null);
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    canvas.getContext("2d").drawImage(this.video, 0, 0, 640, 480);
    return new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.85));
  }

  calibrate() {
    if (this.mode !== "live" || this.video.readyState < 2) return false;
    this.analysisCtx.drawImage(this.video, 0, 0, ANALYSIS_W, ANALYSIS_H);
    this.background = this.analysisCtx.getImageData(0, 0, ANALYSIS_W, ANALYSIS_H).data;
    this.box = null;
    return true;
  }

  // El brazo es naranja intenso; sus píxeles no cuentan como objeto.
  static isRobotColor(r, g, b) {
    return r > 140 && r - b > 90 && g < r * 0.78 && g > r * 0.22;
  }

  // Busca el objeto más grande que ha aparecido respecto a la mesa calibrada, ignorando el brazo.
  detect() {
    const W = ANALYSIS_W, H = ANALYSIS_H, N = W * H;
    this.analysisCtx.drawImage(this.video, 0, 0, W, H);
    const pixels = this.analysisCtx.getImageData(0, 0, W, H).data;
    const changed = new Uint8Array(N);
    const robot = new Uint8Array(N);
    for (let p = 0, i = 0; p < N; p++, i += 4) {
      const difference = (Math.abs(pixels[i] - this.background[i]) + Math.abs(pixels[i + 1] - this.background[i + 1]) + Math.abs(pixels[i + 2] - this.background[i + 2])) / 3;
      if (difference > this.threshold) changed[p] = 1;
      if (CameraView.isRobotColor(pixels[i], pixels[i + 1], pixels[i + 2])) robot[p] = 1;
    }
    // Se ensancha la zona del brazo para quitar también sus bordes y cables.
    const R = 3;
    const blocked = new Uint8Array(N);
    for (let p = 0; p < N; p++) {
      if (!robot[p]) continue;
      const x = p % W, y = (p / W) | 0;
      for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
        const nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < W && ny >= 0 && ny < H) blocked[ny * W + nx] = 1;
      }
    }
    const seen = new Uint8Array(N);
    const stack = new Int32Array(N);
    let best = null;
    for (let start = 0; start < N; start++) {
      if (!changed[start] || blocked[start] || seen[start]) continue;
      let top = 0, count = 0, minX = W, minY = H, maxX = 0, maxY = 0;
      stack[top++] = start;
      seen[start] = 1;
      while (top) {
        const p = stack[--top];
        const x = p % W, y = (p / W) | 0;
        count++;
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        for (const q of [x > 0 ? p - 1 : -1, x < W - 1 ? p + 1 : -1, y > 0 ? p - W : -1, y < H - 1 ? p + W : -1]) {
          if (q >= 0 && changed[q] && !blocked[q] && !seen[q]) { seen[q] = 1; stack[top++] = q; }
        }
      }
      if (!best || count > best.count) best = { count, minX, minY, maxX, maxY };
    }
    const sx = this.canvas.width / W;
    const sy = this.canvas.height / H;
    if (!best || best.count * sx * sy < this.minPixelArea || best.maxX <= best.minX || best.maxY <= best.minY) return null;
    return { x: best.minX * sx, y: best.minY * sy, w: (best.maxX - best.minX + 1) * sx, h: (best.maxY - best.minY + 1) * sy };
  }
}
