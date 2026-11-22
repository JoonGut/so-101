// Prueba el análisis de imagen sin navegador, cámaras ni dependencias externas.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const sandbox = vm.createContext({ performance });
vm.runInContext(`${fs.readFileSync("static/js/vision.js", "utf8")}\nthis.CameraView = CameraView;`, sandbox);

// Se crea sin constructor para no necesitar DOM.
function camera(frame) {
  return Object.assign(Object.create(sandbox.CameraView.prototype), {
    canvas: { width: 640, height: 480 },
    video: { readyState: 2 },
    mode: "live",
    threshold: 30,
    minPixelArea: 3500,
    requestId: 0,
    analysisCtx: { drawImage() {}, getImageData: () => ({ data: frame.slice() }) },
  });
}

function paint(frame, x0, x1, y0, y1, [r, g, b]) {
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) frame.set([r, g, b, 255], (y * 160 + x) * 4);
}

test("calibración y detección usan el mismo tamaño de imagen", () => {
  const frame = new Uint8ClampedArray(160 * 120 * 4);
  const view = camera(frame);
  assert.equal(view.calibrate(), true);
  assert.equal(view.background.length, frame.length);
  assert.equal(view.detect(), null);
  paint(frame, 40, 70, 30, 60, [255, 255, 255]);
  const box = view.detect();
  assert.deepEqual([box.x, box.y, box.w, box.h], [160, 120, 120, 120]);
});

test("el brazo naranja no se detecta y no se une al objeto que tiene al lado", () => {
  const frame = new Uint8ClampedArray(160 * 120 * 4);
  paint(frame, 0, 160, 0, 120, [30, 30, 30]); // tapete negro
  const view = camera(frame);
  view.calibrate();
  paint(frame, 90, 140, 40, 90, [235, 110, 30]); // brazo naranja que se ha movido
  assert.equal(view.detect(), null);
  paint(frame, 70, 86, 50, 70, [60, 80, 200]); // lata azul pegada al brazo
  const box = view.detect();
  assert.ok(box, "debe encontrar la lata");
  assert.ok(box.x >= 70 * 4 && box.x + box.w <= 87 * 4, `el recuadro no incluye el brazo: ${JSON.stringify(box)}`);
});

test("no se calibra sin vídeo en directo", () => {
  const view = camera(new Uint8ClampedArray(160 * 120 * 4));
  view.mode = "model";
  assert.equal(view.calibrate(), false);
});

test("al soltar la cámara para el modelo se descarta una conexión pendiente", async () => {
  let resolveCamera;
  let stopped = false;
  sandbox.navigator = { mediaDevices: {
    enumerateDevices: async () => [{ kind: "videoinput", deviceId: "ext", label: "Wed Camera" }],
    getUserMedia: () => new Promise(resolve => resolveCamera = resolve),
  } };
  const view = Object.assign(camera(new Uint8ClampedArray(4)), {
    preferredCamera: /We[bd] Camera/i,
    stopStream() {}, showPreview() {}, setBadge() {}, onStatus() {},
    preview: {}, detectionLabel: null,
  });
  const request = view.connect();
  await new Promise(resolve => setImmediate(resolve));
  view.release();
  resolveCamera({ getTracks: () => [{ stop: () => stopped = true }] });
  assert.equal(await request, false);
  assert.equal(stopped, true);
  assert.equal(view.mode, "model");
});
