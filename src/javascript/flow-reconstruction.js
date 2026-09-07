/*
 * flow-reconstruction.js
 *
 * Browser demo of the masked ViT / t-ViT autoencoders from the thesis: the
 * user picks which patches of a cylinder-wake snapshot the model is allowed
 * to see, and the model reconstructs the full (u, v) field from them.
 *
 * Mirrors web_demo/demo.py. The models were exported to int8 ONNX by
 * web_demo/export_onnx.py, which also cut the validation window down to the
 * strided snapshots fetched here; the t-ViT's history is the preceding
 * th_max entries of that strided array, since the stride is its dt_h.
 */

(function () {
  "use strict";

  const DATA_DIR = "assets/data";
  const MODEL_DIR = "assets/models";
  const MODELS = {
    vit: { path: `${MODEL_DIR}/vit-flow.onnx`, temporal: false },
    tvit: { path: `${MODEL_DIR}/tvit-flow.onnx`, temporal: true }
  };

  // Masked patches are painted this colour, standing in for matplotlib's
  // set_bad("lightgray") in demo.py
  const MASKED_RGB = [38, 38, 58];

  const VIRIDIS = [
    [68, 1, 84], [72, 33, 115], [67, 62, 133], [56, 88, 140], [45, 112, 142],
    [37, 133, 142], [30, 155, 138], [53, 183, 121], [109, 205, 89],
    [180, 222, 44], [253, 231, 37]
  ];
  const PIYG = [
    [142, 1, 82], [197, 27, 125], [222, 119, 174], [241, 182, 218],
    [253, 224, 239], [247, 247, 247], [230, 245, 208], [184, 225, 134],
    [127, 188, 65], [77, 146, 33], [39, 100, 25]
  ];

  const root = document.getElementById("flow-demo");
  if (!root) {
    return;
  }

  const el = {
    status: document.getElementById("flow-status"),
    frame: document.getElementById("flow-frame"),
    frameLabel: document.getElementById("flow-frame-label"),
    field: document.getElementById("flow-field"),
    play: document.getElementById("flow-play"),
    count: document.getElementById("flow-count"),
    error: document.getElementById("flow-error"),
    modelButtons: root.querySelectorAll("[data-model]"),
    maskButtons: root.querySelectorAll("[data-mask]")
  };

  const panels = ["target", "input", "output"].reduce((acc, name) => {
    const canvas = document.getElementById(`flow-${name}`);
    acc[name] = { canvas, ctx: canvas.getContext("2d") };
    return acc;
  }, {});

  const state = {
    manifest: null,
    frames: null,       // Float32Array, (n_frames, H, W, C)
    unmasked: new Set(),
    modelName: "vit",
    sessions: {},       // lazily created InferenceSessions
    frameIdx: 0,
    field: "vorticity",
    recon: null,        // Float32Array (H*W*C) in physical units, or null
    running: false,
    queued: false,
    playing: false
  };

  // Pause between snapshots during playback, on top of however long the
  // model takes - each step waits for its own inference rather than firing on
  // a fixed timer, so playback can't outrun the model
  const PLAY_DELAY = 120;

  /* ------------------------------------------------------------------ */
  /* Field helpers                                                       */
  /* ------------------------------------------------------------------ */

  function frameView(f) {
    const { H, W, C } = state.manifest;
    const size = H * W * C;
    return state.frames.subarray(f * size, (f + 1) * size);
  }

  function normalize(U) {
    const { mean_u, std_u, mean_v, std_v } = state.manifest.norm_stats;
    const out = new Float32Array(U.length);
    for (let i = 0; i < U.length; i += 2) {
      out[i] = (U[i] - mean_u) / std_u;
      out[i + 1] = (U[i + 1] - mean_v) / std_v;
    }
    return out;
  }

  function denormalize(X) {
    const { mean_u, std_u, mean_v, std_v } = state.manifest.norm_stats;
    const out = new Float32Array(X.length);
    for (let i = 0; i < X.length; i += 2) {
      out[i] = X[i] * std_u + mean_u;
      out[i + 1] = X[i + 1] * std_v + mean_v;
    }
    return out;
  }

  /* omega_z = dv/dx - du/dy, central differences with one-sided edges, the
     same convention as numpy's gradient in data.compute_vorticity */
  function vorticity(U) {
    const { H, W, dx, dy } = state.manifest;
    const out = new Float32Array(H * W);
    const u = (y, x) => U[(y * W + x) * 2];
    const v = (y, x) => U[(y * W + x) * 2 + 1];

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let du_dy;
        if (y === 0) du_dy = (u(1, x) - u(0, x)) / dy;
        else if (y === H - 1) du_dy = (u(H - 1, x) - u(H - 2, x)) / dy;
        else du_dy = (u(y + 1, x) - u(y - 1, x)) / (2 * dy);

        let dv_dx;
        if (x === 0) dv_dx = (v(y, 1) - v(y, 0)) / dx;
        else if (x === W - 1) dv_dx = (v(y, W - 1) - v(y, W - 2)) / dx;
        else dv_dx = (v(y, x + 1) - v(y, x - 1)) / (2 * dx);

        out[y * W + x] = dv_dx - du_dy;
      }
    }
    return out;
  }

  /* Pull out the scalar the user asked to see, as an (H*W) array */
  function scalarField(U) {
    const { H, W } = state.manifest;
    if (state.field === "vorticity") {
      return vorticity(U);
    }

    const offset = state.field === "u" ? 0 : 1;
    const out = new Float32Array(H * W);
    for (let i = 0; i < H * W; i++) {
      out[i] = U[i * 2 + offset];
    }
    return out;
  }

  /* Colour limits, taken from the target so all three panels share a scale.
     Vorticity uses +/-3 sigma, matching demo.py's plot_fields. */
  function colourLimits(target) {
    if (state.field === "vorticity") {
      let sum = 0;
      for (let i = 0; i < target.length; i++) sum += target[i];
      const mean = sum / target.length;

      let acc = 0;
      for (let i = 0; i < target.length; i++) {
        acc += (target[i] - mean) * (target[i] - mean);
      }
      const vmax = 3 * Math.sqrt(acc / target.length);
      return { vmin: -vmax, vmax, cmap: PIYG };
    }

    let vmin = Infinity;
    let vmax = -Infinity;
    for (let i = 0; i < target.length; i++) {
      if (target[i] < vmin) vmin = target[i];
      if (target[i] > vmax) vmax = target[i];
    }
    return { vmin, vmax, cmap: VIRIDIS };
  }

  function sampleCmap(cmap, t) {
    if (!(t >= 0)) t = 0;
    else if (t > 1) t = 1;

    const pos = t * (cmap.length - 1);
    const i = Math.min(cmap.length - 2, Math.floor(pos));
    const f = pos - i;
    const a = cmap[i];
    const b = cmap[i + 1];
    return [
      a[0] + (b[0] - a[0]) * f,
      a[1] + (b[1] - a[1]) * f,
      a[2] + (b[2] - a[2]) * f
    ];
  }

  /* ------------------------------------------------------------------ */
  /* Drawing                                                             */
  /* ------------------------------------------------------------------ */

  function draw(panel, scalar, limits, maskHidden) {
    const { H, W, P, w } = state.manifest;
    const { ctx, canvas } = panel;
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width = W;
      canvas.height = H;
    }

    const image = ctx.createImageData(W, H);
    const px = image.data;
    const span = limits.vmax - limits.vmin || 1;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const o = i * 4;

        const patch = Math.floor(y / P) * w + Math.floor(x / P);
        if (maskHidden && !state.unmasked.has(patch)) {
          px[o] = MASKED_RGB[0];
          px[o + 1] = MASKED_RGB[1];
          px[o + 2] = MASKED_RGB[2];
        } else {
          const [r, g, b] = sampleCmap(
            limits.cmap, (scalar[i] - limits.vmin) / span);
          px[o] = r;
          px[o + 1] = g;
          px[o + 2] = b;
        }
        px[o + 3] = 255;
      }
    }

    ctx.putImageData(image, 0, 0);
  }

  function drawPending(panel) {
    const { ctx, canvas } = panel;
    ctx.fillStyle = `rgb(${MASKED_RGB.join(",")})`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  function render() {
    const U_target = frameView(state.frameIdx);
    const target = scalarField(U_target);
    const limits = colourLimits(target);

    draw(panels.target, target, limits, false);
    draw(panels.input, target, limits, true);

    if (state.recon) {
      draw(panels.output, scalarField(state.recon), limits, false);
    } else {
      drawPending(panels.output);
    }

    el.count.textContent = `${state.unmasked.size} / ${state.manifest.h * state.manifest.w}`;
  }

  /* ------------------------------------------------------------------ */
  /* Patch <-> tensor plumbing                                           */
  /* ------------------------------------------------------------------ */

  /* Gather the unmasked patches of one normalised frame into the flat
     [N_un, C, P, P] block the model expects (data.patchify, restricted to
     I_unmasked), writing at `offset` into `out` */
  function writePatches(out, offset, X, indices) {
    const { W, C, P, w } = state.manifest;
    let k = offset;

    for (const n of indices) {
      const y0 = Math.floor(n / w) * P;
      const x0 = (n % w) * P;

      for (let c = 0; c < C; c++) {
        for (let i = 0; i < P; i++) {
          const row = (y0 + i) * W;
          for (let j = 0; j < P; j++) {
            out[k++] = X[(row + x0 + j) * C + c];
          }
        }
      }
    }
  }

  /* The decoder reconstructs every patch, including the ones it was handed.
     Showing its version of those would be misleading - only the masked
     regions are actually predictions - so the visible patches are pasted
     back over the output. */
  function pasteVisible(recon, U_target, indices) {
    const { W, C, P, w } = state.manifest;

    for (const n of indices) {
      const y0 = Math.floor(n / w) * P;
      const x0 = (n % w) * P;

      for (let i = 0; i < P; i++) {
        const row = (y0 + i) * W;
        for (let j = 0; j < P; j++) {
          const base = (row + x0 + j) * C;
          for (let c = 0; c < C; c++) {
            recon[base + c] = U_target[base + c];
          }
        }
      }
    }
    return recon;
  }

  /* Inverse of data.patchify over all N patches: [N, C, P, P] -> (H, W, C) */
  function unpatchify(patches) {
    const { H, W, C, P, h, w } = state.manifest;
    const out = new Float32Array(H * W * C);

    for (let n = 0; n < h * w; n++) {
      const y0 = Math.floor(n / w) * P;
      const x0 = (n % w) * P;
      const base = n * C * P * P;

      for (let c = 0; c < C; c++) {
        for (let i = 0; i < P; i++) {
          const row = (y0 + i) * W;
          for (let j = 0; j < P; j++) {
            out[(row + x0 + j) * C + c] = patches[base + (c * P + i) * P + j];
          }
        }
      }
    }
    return out;
  }

  /* ------------------------------------------------------------------ */
  /* Inference                                                           */
  /* ------------------------------------------------------------------ */

  async function getSession(name) {
    if (state.sessions[name]) {
      return state.sessions[name];
    }

    setStatus(`Loading the ${label(name)} (~25 MB, first time only)...`);
    const session = await ort.InferenceSession.create(MODELS[name].path, {
      executionProviders: ["wasm"]
    });
    state.sessions[name] = session;
    return session;
  }

  async function runModel() {
    // Coalesce clicks made while a run is in flight
    if (state.running) {
      state.queued = true;
      return;
    }

    const indices = [...state.unmasked].sort((a, b) => a - b);
    if (!indices.length) {
      state.recon = null;
      setStatus("Unmask at least one patch for the model to work from.");
      render();
      return;
    }

    state.running = true;
    setError("");

    try {
      const { C, P, th_max } = state.manifest;
      const { temporal } = MODELS[state.modelName];
      const session = await getSession(state.modelName);

      setStatus("Reconstructing...");

      // Pinned for the whole run, since the slider can move while it's in
      // flight - the queued rerun picks up any change
      const frameIdx = state.frameIdx;

      const n_un = indices.length;
      const perFrame = n_un * C * P * P;
      const steps = temporal ? th_max : 1;

      const input = new Float32Array(steps * perFrame);
      for (let t = 0; t < steps; t++) {
        // The t-ViT sees th_max snapshots dt_h apart ending on the selected
        // one; the shipped frames are already strided by dt_h
        const f = temporal ? frameIdx - (steps - 1 - t) : frameIdx;
        writePatches(input, t * perFrame, normalize(frameView(f)), indices);
      }

      const dims = temporal
        ? [1, steps, n_un, C, P, P]
        : [1, n_un, C, P, P];

      const feeds = {
        Xpatch_unmasked: new ort.Tensor("float32", input, dims),
        I_patch: new ort.Tensor(
          "int64", BigInt64Array.from(indices, BigInt), [1, n_un])
      };

      const results = await session.run(feeds);
      const out = results[session.outputNames[0]];

      // [1, N, C, P, P], or [1, Th, N, C, P, P] where the last step is current
      const patchCount = state.manifest.h * state.manifest.w;
      const stride = patchCount * C * P * P;
      const start = temporal ? (steps - 1) * stride : 0;
      const recon = out.data.subarray(start, start + stride);

      state.recon = pasteVisible(
        denormalize(unpatchify(recon)), frameView(frameIdx), indices);
      setStatus(`Reconstructed from ${n_un} of ${patchCount} patches`
        + ` using the ${label(state.modelName)}.`);
    } catch (err) {
      console.error(err);
      state.recon = null;
      setError(`Could not run the model: ${err.message}`);
      setStatus("");
      setPlaying(false);   // don't keep looping on a broken model
    } finally {
      state.running = false;
      render();

      if (state.queued) {
        state.queued = false;
        runModel();
      }
    }
  }

  function label(name) {
    return name === "tvit" ? "t-ViT" : "ViT";
  }

  function setStatus(text) {
    el.status.textContent = text;
  }

  function setError(text) {
    el.error.textContent = text;
    el.error.hidden = !text;
  }

  /* ------------------------------------------------------------------ */
  /* Interaction                                                         */
  /* ------------------------------------------------------------------ */

  function patchAt(event) {
    const { h, w } = state.manifest;
    const rect = panels.input.canvas.getBoundingClientRect();
    const col = Math.floor((event.clientX - rect.left) / rect.width * w);
    const row = Math.floor((event.clientY - rect.top) / rect.height * h);

    if (col < 0 || col >= w || row < 0 || row >= h) {
      return -1;
    }
    return row * w + col;
  }

  function bindInput() {
    const canvas = panels.input.canvas;
    let painting = false;
    let paintTo = true;      // whether this drag is unmasking or masking
    let lastPatch = -1;

    canvas.addEventListener("pointerdown", (event) => {
      const n = patchAt(event);
      if (n < 0) return;

      event.preventDefault();
      canvas.setPointerCapture(event.pointerId);
      painting = true;
      paintTo = !state.unmasked.has(n);   // drag continues what the click started
      lastPatch = n;
      applyPatch(n, paintTo);
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!painting) return;
      const n = patchAt(event);
      if (n < 0 || n === lastPatch) return;
      lastPatch = n;
      applyPatch(n, paintTo);
    });

    const release = (event) => {
      if (!painting) return;
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
      painting = false;
      lastPatch = -1;
      runModel();
    };

    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
  }

  function applyPatch(n, on) {
    if (on) state.unmasked.add(n);
    else state.unmasked.delete(n);
    render();
  }

  /* ------------------------------------------------------------------ */
  /* Playback                                                            */
  /* ------------------------------------------------------------------ */

  function advanceFrame() {
    // Read the bounds off the slider, since the minimum moves when the model
    // changes (the t-ViT needs history)
    const min = Number(el.frame.min);
    const max = Number(el.frame.max);

    state.frameIdx = state.frameIdx >= max ? min : state.frameIdx + 1;
    el.frame.value = String(state.frameIdx);
    updateFrameLabel();
  }

  async function playLoop() {
    while (state.playing) {
      advanceFrame();
      // The previous reconstruction stays on screen until the new one lands,
      // rather than blanking between every frame
      render();
      await runModel();

      if (!state.playing) {
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, PLAY_DELAY));
    }
  }

  function setPlaying(on) {
    if (on === state.playing) {
      return;
    }

    state.playing = on;
    el.play.innerHTML = on ? "&#9208;" : "&#9654;";
    el.play.setAttribute("aria-label", on ? "Pause" : "Play");
    el.play.classList.toggle("is-active", on);

    if (on) {
      playLoop();
    }
  }

  function setMask(preset) {
    const { h, w } = state.manifest;
    state.unmasked.clear();

    if (preset === "default") {
      // The 25%-unmasked pattern used at training time (I_UNMASKED in demo.py)
      for (const n of [2, 3, 10, 11, 18, 19, 26, 27]) state.unmasked.add(n);
    } else if (preset === "all") {
      for (let n = 0; n < h * w; n++) state.unmasked.add(n);
    }
    // "none" leaves it empty

    render();
    runModel();
  }

  /* ------------------------------------------------------------------ */
  /* Setup                                                               */
  /* ------------------------------------------------------------------ */

  function bindControls() {
    el.modelButtons.forEach((button) => {
      button.addEventListener("click", () => {
        state.modelName = button.dataset.model;
        el.modelButtons.forEach((b) => {
          b.classList.toggle("is-active", b === button);
        });
        updateFrameBounds();
        runModel();
      });
    });

    el.maskButtons.forEach((button) => {
      button.addEventListener("click", () => setMask(button.dataset.mask));
    });

    el.field.addEventListener("change", () => {
      state.field = el.field.value;
      render();
    });

    el.play.addEventListener("click", () => setPlaying(!state.playing));

    el.frame.addEventListener("input", () => {
      setPlaying(false);   // scrubbing takes over from playback
      state.frameIdx = Number(el.frame.value);
      updateFrameLabel();
      state.recon = null;
      render();
    });
    el.frame.addEventListener("change", () => runModel());
  }

  /* The t-ViT needs th_max-1 snapshots of history, so the earliest frames are
     only selectable for the plain ViT */
  function updateFrameBounds() {
    const { n_frames, th_max } = state.manifest;
    const min = MODELS[state.modelName].temporal ? th_max - 1 : 0;

    el.frame.min = String(min);
    el.frame.max = String(n_frames - 1);
    if (state.frameIdx < min) {
      state.frameIdx = min;
      state.recon = null;
    }
    el.frame.value = String(state.frameIdx);
    updateFrameLabel();
  }

  function updateFrameLabel() {
    const source = state.manifest.source_indices[state.frameIdx];
    el.frameLabel.textContent = `snapshot ${source}`;
  }

  async function main() {
    try {
      setStatus("Loading the flow data...");

      const [manifest, buffer] = await Promise.all([
        fetch(`${DATA_DIR}/flow-window.json`).then((r) => {
          if (!r.ok) throw new Error(`flow-window.json: ${r.status}`);
          return r.json();
        }),
        fetch(`${DATA_DIR}/flow-window.bin`).then((r) => {
          if (!r.ok) throw new Error(`flow-window.bin: ${r.status}`);
          return r.arrayBuffer();
        })
      ]);

      state.manifest = manifest;
      state.frames = new Float32Array(buffer);
      state.frameIdx = manifest.th_max - 1;

      bindControls();
      bindInput();
      updateFrameBounds();
      setMask("default");
    } catch (err) {
      console.error(err);
      setStatus("");
      setError(`Could not load the demo data: ${err.message}.`
        + " The model and data files may not have been exported yet"
        + " (see tmp/web_demo/export_onnx.py).");
    }
  }

  if (typeof ort !== "undefined") {
    // Match the CDN the runtime itself is loaded from in base.njk, so the
    // wasm binaries resolve against it rather than this site
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/";
  }

  main();
})();
