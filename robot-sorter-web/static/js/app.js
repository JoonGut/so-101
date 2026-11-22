// Estación de recogida: 1 calibrar mesa · 2 elegir modelo · 3 recoger · 4 ajustar.
const $ = id => document.getElementById(id);

async function api(path, { method = "GET", body } = {}) {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Error ${response.status}`);
  return data;
}

function setStatus(element, text, kind = "") {
  element.textContent = text;
  element.className = `status ${kind}`;
}

// ---------------------------------------------------------------- Cámaras
const front = new CameraView($("cam-front"), /We[bd] Camera/i, { detect: true });
const wrist = new CameraView($("cam-wrist"), /USB2\.0[_ ]?CAM/i);
const cameras = [front, wrist];

// Una detrás de otra: así el navegador pide permiso una sola vez.
async function connectCameras() {
  for (const camera of cameras) await camera.connect();
}

// ---------------------------------------------------------------- 1. Calibrar mesa
const calibrateButton = $("btn-calibrate");
const calibrateStatus = $("calibrate-status");

function renderCalibration() {
  calibrateButton.disabled = front.mode !== "live";
  if (front.mode === "model") setStatus(calibrateStatus, "La cámara la está usando el modelo.");
  else if (front.mode === "offline") setStatus(calibrateStatus, front.error || "Sin cámara frontal.", "warn");
  else if (front.background) setStatus(calibrateStatus, "✓ Mesa calibrada. Los objetos que pongas aparecerán recuadrados.", "ok");
  else setStatus(calibrateStatus, "Cámara lista. Mesa sin calibrar.");
}
front.onStatus = renderCalibration;
calibrateButton.addEventListener("click", () => {
  if (front.calibrate()) renderCalibration();
});

// ---------------------------------------------------------------- 1b. Alinear la cámara
const align = $("align");
const alignStatus = $("align-status");
const alignInput = $("in-align");
const reference = new Image();
let aligning = false;

alignInput.addEventListener("input", () => {
  front.overlayAlpha = Number(alignInput.value);
  $("out-align").textContent = `${Math.round(alignInput.value * 100)} %`;
});

async function measureAlignment() {
  if (!align.open || aligning) return;
  aligning = true;
  try {
    const frame = await front.snapshot();
    if (!frame) {
      setStatus(alignStatus, "Hace falta la cámara frontal en directo.", "warn");
      return;
    }
    const response = await fetch("/api/reference/compare", { method: "POST", body: frame });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Error ${response.status}`);
    setStatus(alignStatus, data.message, data.aligned ? "ok" : "warn");
    alignStatus.classList.add("align-status-big");
  } catch (error) {
    setStatus(alignStatus, error.message, "warn");
  } finally {
    aligning = false;
  }
}

align.addEventListener("toggle", async () => {
  if (!align.open) {
    front.overlay = null;
    return;
  }
  const info = await api("/api/reference").catch(() => ({ available: false }));
  if (!info.available) {
    setStatus(alignStatus, "Falta la referencia: ejecuta ./comprobar_camara.sh en la carpeta del proyecto.", "warn");
    return;
  }
  reference.src = `/api/reference/front.jpg?t=${Date.now()}`;
  await reference.decode().catch(() => {});
  front.overlay = reference;
  front.overlayAlpha = Number(alignInput.value);
  measureAlignment();
});
setInterval(measureAlignment, 1000);

// ---------------------------------------------------------------- 2. Modelos
const modelList = $("model-list");
const modelStatus = $("model-status");
let running = false;

function renderModels({ models = [], active }) {
  modelList.replaceChildren();
  if (!models.length) {
    modelList.innerHTML = "<li>No hay modelos. Sube uno.</li>";
    return;
  }
  for (const model of models) {
    const item = document.createElement("li");
    item.classList.toggle("active", model.id === active);
    const label = document.createElement("label");
    const radio = Object.assign(document.createElement("input"), {
      type: "radio", name: "model", checked: model.id === active, disabled: !model.compatible || running,
    });
    radio.addEventListener("change", async () => {
      try {
        renderModels(await api("/api/models/select", { method: "POST", body: { id: model.id } }));
        setStatus(modelStatus, `Modelo elegido: ${model.name}.`, "ok");
        refreshRun();
      } catch (error) {
        setStatus(modelStatus, error.message, "warn");
        loadModels();
      }
    });
    const text = document.createElement("span");
    text.textContent = model.name;
    const details = document.createElement("small");
    details.textContent = model.compatible
      ? `${model.type} · cámaras ${model.cameras.join(" + ")} · ${model.size_mb} MB`
      : `No compatible: ${model.problems.join("; ")}`;
    if (!model.compatible) details.className = "problem";
    text.append(details);
    label.append(radio, text);
    item.append(label);
    if (!model.builtin) {
      const remove = Object.assign(document.createElement("button"), { type: "button", textContent: "Borrar" });
      remove.disabled = running;
      remove.addEventListener("click", async () => {
        // Segunda pulsación para confirmar (sin diálogos que bloqueen la página).
        if (!remove.dataset.armed) {
          remove.dataset.armed = "1";
          remove.textContent = "¿Seguro?";
          setTimeout(() => { delete remove.dataset.armed; remove.textContent = "Borrar"; }, 4000);
          return;
        }
        try {
          renderModels(await api(`/api/models/${encodeURIComponent(model.id)}`, { method: "DELETE" }));
          setStatus(modelStatus, `Modelo «${model.name}» borrado.`);
          refreshRun();
        } catch (error) {
          setStatus(modelStatus, error.message, "warn");
        }
      });
      item.append(remove);
    }
    modelList.append(item);
  }
}

async function loadModels() {
  try {
    renderModels(await api("/api/models"));
  } catch (error) {
    setStatus(modelStatus, error.message, "warn");
  }
}

$("upload-form").addEventListener("submit", event => {
  event.preventDefault();
  const file = $("upload-file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("name", $("upload-name").value);
  const progress = $("upload-progress");
  const button = $("upload-btn");
  progress.hidden = false;
  progress.value = 0;
  button.disabled = true;
  setStatus(modelStatus, `Subiendo ${file.name}…`);
  // XMLHttpRequest para poder mostrar el progreso.
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/models/upload");
  xhr.upload.onprogress = e => {
    if (!e.lengthComputable) return;
    progress.value = e.loaded / e.total * 100;
    if (e.loaded === e.total) setStatus(modelStatus, "Comprobando el modelo…");
  };
  const done = () => { progress.hidden = true; button.disabled = false; };
  xhr.onload = () => {
    done();
    let result = {};
    try { result = JSON.parse(xhr.responseText); } catch { /* no JSON */ }
    if (xhr.status >= 200 && xhr.status < 300) {
      renderModels(result);
      $("upload-form").reset();
      setStatus(modelStatus, result.compatible
        ? `«${result.name}» subido. Selecciónalo para usarlo.`
        : `«${result.name}» subido, pero no es compatible: ${result.problems.join("; ")}.`, result.compatible ? "ok" : "warn");
    } else {
      setStatus(modelStatus, typeof result.detail === "string" ? result.detail : `Error al subir (${xhr.status}).`, "warn");
    }
  };
  xhr.onerror = () => { done(); setStatus(modelStatus, "Se ha perdido la conexión durante la subida.", "warn"); };
  xhr.send(form);
});

// ---------------------------------------------------------------- 3. Recoger basura
const playButton = $("btn-play");
const stopButton = $("btn-stop");
const estopButton = $("btn-estop");
const runStatus = $("run-status");
const runLog = $("run-log");
let runError = null;
let stopping = false;

async function play() {
  playButton.disabled = true;
  runError = null;
  // Las cámaras pasan al modelo: el navegador las suelta antes.
  cameras.forEach(camera => camera.release());
  setStatus(runStatus, "Preparando el brazo y cargando el modelo…");
  await new Promise(resolve => setTimeout(resolve, 800));
  try {
    const auto = $("run-auto").checked;
    render(await api("/api/policy/start", { method: "POST", body: {
      duration: Number($("run-duration").value), task: $("run-task").value, auto,
    } }));
  } catch (error) {
    runError = error.message;
    connectCameras();
    refreshRun();
  }
}

async function stop() {
  stopping = true;
  stopButton.disabled = true;
  setStatus(runStatus, "Parando: el brazo vuelve a su punto de partida…");
  try {
    render(await api("/api/policy/stop", { method: "POST" }));
  } catch (error) {
    runError = error.message;
  } finally {
    stopping = false;
  }
}

async function emergencyStop() {
  try {
    await api("/api/estop", { method: "POST", body: { active: true } });
    runError = "Parada de emergencia: el brazo se ha quedado quieto donde estaba.";
  } catch (error) {
    runError = error.message;
  }
  refreshRun();
}

playButton.addEventListener("click", play);
stopButton.addEventListener("click", stop);
estopButton.addEventListener("click", emergencyStop);

function render(data) {
  const wasRunning = running;
  running = data.state === "preparing" || data.state === "running";
  if (running && !wasRunning) cameras.forEach(camera => camera.mode !== "model" && camera.release());
  if (!running && wasRunning) {
    connectCameras();
    loadParams();  // la corrección encontrada por el modelo se refleja en el control
  }
  if (running !== wasRunning) loadModels();

  const ready = data.model?.compatible && data.calibrated;
  playButton.disabled = running || !ready;
  stopButton.disabled = data.state !== "running" || stopping;
  estopButton.disabled = !running;
  $("run-duration").disabled = running;
  const autoBox = $("run-auto");
  if (data.detector && !data.detector.available) {
    autoBox.checked = false;
    autoBox.disabled = true;
    autoBox.title = "Falta entrenar el detector: ./entrenar_detector.sh";
  } else if (data.detector && autoBox.dataset.init !== "1") {
    autoBox.dataset.init = "1";
    autoBox.checked = true;
    if (Number($("run-duration").value) < 300) $("run-duration").value = "3600";
  }
  autoBox.disabled = running || !data.detector?.available;
  renderAuto(running ? data.auto : null);
  $("task-row").hidden = !data.model || data.model.type === "act" || autoBox.checked;
  $("run-task").disabled = running;

  if (runError && !running) setStatus(runStatus, runError, "warn");
  else if (!data.model) setStatus(runStatus, "Elige o sube un modelo en el paso 2.", "warn");
  else if (!data.model.compatible) setStatus(runStatus, "El modelo elegido no es compatible.", "warn");
  else if (!data.calibrated) setStatus(runStatus, "Falta la calibración de las articulaciones del robot (so101).", "warn");
  else if (stopping) setStatus(runStatus, "Parando: el brazo vuelve a su punto de partida…");
  else {
    const messages = {
      idle: `Listo para recoger con «${data.model.name}».`,
      preparing: "Preparando el brazo y cargando el modelo…",
      running: "Recogiendo… Mantén la mano cerca de la alimentación del brazo.",
      finished: "✓ Terminado. El brazo ha vuelto a su punto de partida.",
      stopped: "Parado. El brazo ha vuelto a su punto de partida.",
      error: "Ha fallado. Abre el registro para ver el motivo.",
    };
    setStatus(runStatus, messages[data.state] || "", data.state === "error" ? "warn" : data.state === "finished" ? "ok" : "");
  }
  if (running) runError = null;

  if (running) {
    const stamp = Date.now();
    if (data.previews.includes("front")) front.updatePreview(`/api/policy/preview/front.jpg?t=${stamp}`);
    if (data.previews.includes("gripper")) wrist.updatePreview(`/api/policy/preview/gripper.jpg?t=${stamp}`);
  }
  if (runLog.textContent !== data.output) {
    const follow = runLog.scrollHeight - runLog.scrollTop - runLog.clientHeight < 40;
    runLog.textContent = data.output;
    if (follow) runLog.scrollTop = runLog.scrollHeight;
  }
}

function renderAuto(auto) {
  const panel = $("auto-panel");
  panel.hidden = !auto;
  if (!auto) {
    front.detectionLabel.textContent = "Objeto en la mesa";
    return;
  }
  const phases = { observing: "Mirando la mesa…", running: "Ejecutando:", returning: "Volviendo a la base…" };
  $("auto-phase").textContent = phases[auto.phase] || "";
  let detail = "";
  if (auto.phase === "observing") {
    const seen = auto.seen === "Nada" ? "mesa recogida: el brazo espera quieto" : `ve «${auto.seen}»`;
    $("auto-instruction").textContent = "";
    detail = `La cámara ${seen} (${Math.round((auto.confidence || 0) * 100)} %)`;
    if (auto.seen && auto.seen !== "Nada") detail += ` · confirmando ${auto.stable}/${auto.needed}`;
    if (auto.objects?.length) detail += ` · fuera: ${auto.objects.join(", ")}`;
  } else {
    $("auto-instruction").textContent = auto.instruction || "";
  }
  if (auto.episodes) detail += `${detail ? " · " : ""}${auto.episodes} instrucción(es) completada(s)`;
  $("auto-detail").textContent = detail;
  front.detectionLabel.hidden = !(auto.instruction || (auto.seen && auto.seen !== "Nada"));
  front.detectionLabel.textContent = auto.instruction ? `▶ ${auto.instruction}` : `Ve: ${auto.seen}`;
}

let polling = false;
async function refreshRun() {
  if (polling) return;
  polling = true;
  try {
    render(await api("/api/policy"));
    $("server-dot").className = "dot ok";
    $("server-text").textContent = "Servidor conectado";
  } catch {
    $("server-dot").className = "dot bad";
    $("server-text").textContent = "Sin conexión con el servidor";
  } finally {
    polling = false;
  }
}

// ---------------------------------------------------------------- 4. Ajustes
const DEFAULTS = { speed_factor: 1.0, smoothness_factor: 1, max_acceleration: 2.5, pan_offset: 0, auto_correct: false };
const sliders = [
  { input: $("in-speed"), output: $("out-speed"), format: v => `${v.toFixed(1)}×` },
  { input: $("in-smooth"), output: $("out-smooth"), format: v => `${v} / 10` },
  { input: $("in-accel"), output: $("out-accel"), format: v => `${Math.round(v * 6)}° por paso` },
  { input: $("in-pan"), output: $("out-pan"), format: v => `${v > 0 ? "+" : ""}${v.toFixed(1)}°` },
];
const autoCorrect = $("in-auto");
autoCorrect.addEventListener("change", () => sendParams({ auto_correct: autoCorrect.checked }));

function showValue({ input, output, format }) {
  output.textContent = format(Number(input.value));
}

async function sendParams(params) {
  try {
    await api("/api/params", { method: "POST", body: params });
  } catch (error) {
    setStatus(runStatus, `No se ha podido aplicar el ajuste: ${error.message}`, "warn");
  }
}

for (const slider of sliders) {
  let lastSent = 0;
  const key = slider.input.dataset.key;
  // Se envía mientras se arrastra (como mucho cada 150 ms) y al soltar.
  slider.input.addEventListener("input", () => {
    showValue(slider);
    const now = performance.now();
    if (now - lastSent > 150) {
      lastSent = now;
      sendParams({ [key]: Number(slider.input.value) });
    }
  });
  slider.input.addEventListener("change", () => sendParams({ [key]: Number(slider.input.value) }));
}

function setSliders(params) {
  for (const slider of sliders) {
    const value = params[slider.input.dataset.key];
    if (value !== undefined) slider.input.value = value;
    showValue(slider);
  }
  if (params.auto_correct !== undefined) autoCorrect.checked = params.auto_correct;
}

$("btn-defaults").addEventListener("click", () => {
  setSliders(DEFAULTS);
  sendParams(DEFAULTS);
});

async function loadParams() {
  try {
    setSliders((await api("/api/status")).parameters);
  } catch {
    setSliders(DEFAULTS);
  }
}

// ---------------------------------------------------------------- Arranque
loadParams();
loadModels();
refreshRun().then(() => { if (!running) connectCameras(); });
setInterval(() => { if (!document.hidden) refreshRun(); }, 500);
