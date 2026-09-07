/*
 * fluid-sim.js
 *
 * A port of FluidSimulator (github.com/Ben-Eze/FluidSimulator) to the browser.
 * The classes, methods and variable names mirror the Python: Solver holds the
 * scheme, Fluid holds the fields, GUI handles the brush and Display draws the
 * pxarray. The extras - video recording, data writing, aerofoil BCs, the
 * vorticity/velocity/hue visualisations - are left out.
 *
 * Two differences are forced by the setting:
 *   - The sides are periodic, so instead of numpy's [1:-1, 1:-1] interior
 *     slices every cell is updated through wrapped neighbour indices.
 *   - The fields are flat Float32Arrays indexed as [i_y * Nx + i_x], and since
 *     nothing here allocates a fresh array per operation the way numpy does,
 *     each field carries a companion buffer that it is swapped with.
 */

(function () {
  "use strict";

  // Domain, shared between the spec below and the initial condition. The
  // canvas-wrap aspect-ratio in page.css has to match width : height
  const WIDTH = 192;
  const HEIGHT = 96;
  const BASE_SIZE = 1;

  // Vortex used for the initial condition, in place of the config's uniform
  // freestream - on a periodic domain a freestream would only ever blow the
  // smoke sideways
  const x_c = WIDTH / 2;       // vortex centre
  const y_c = HEIGHT / 2;
  const R = HEIGHT;            // length normalising the vortex profile
  const V_MAX = 12;            // peak swirl speed, about 2 cells per timestep
  const D_MAX = 70;            // peak density of the initial smoke ring (0-255)

  function swirl_profile(x, y) {
    const r2 = ((x - x_c) * (x - x_c) + (y - y_c) * (y - y_c)) / (R * R);
    // r*exp(-6*r^2) peaks at 0.175, which normalises the profile to V_MAX
    return Math.exp(-6 * r2) * V_MAX / (0.175 * R);
  }

  const spec = {
    name: "WebSolver",

    domain: {
      width: WIDTH,
      height: HEIGHT,
      base_size: BASE_SIZE
    },

    time: {
      dt: 1 / 24,
      t_max: Infinity      // the desktop config caps this at 500 for recording
    },

    BCs: {
      sides: {
        top: ["periodic", null],
        bottom: ["periodic", null],
        left: ["periodic", null],
        right: ["periodic", null]
      }
    },

    ICs: {
      U: (x, y) => -(y - y_c) * swirl_profile(x, y),
      V: (x, y) => (x - x_c) * swirl_profile(x, y),
      D: (x, y) => {
        const r_hat = Math.sqrt((x - x_c) * (x - x_c)
          + (y - y_c) * (y - y_c)) / R;
        const ring = r_hat - 0.25;
        return D_MAX * Math.exp(-400 * ring * ring);
      }
    },

    scheme: {
      name: "ImplicitEuler",
      "dx==dy": true,
      nit: 20
    },

    fluid: {
      name: "air",
      viscosity: 1e-2,
      smoke_viscosity: 1e-3,
      smoke_fade: 1
    },

    display: {
      show_smoke: true
    },

    gui: {
      brush_size: 20,
      // The desktop config uses 2, but pygame delivers many more motion events
      // per frame than a browser does, and the strength is split across them
      smoke_strength: 120
    }
  };

  /* ---------------------------------------------------------------------- */

  class InitialCondition {
    constructor(spec, fluid) {
      this.exists = true;
      this.fluid = fluid;

      this.ICspec = spec["ICs"];
      if (!this.ICspec) {
        console.warn("Could not set ICs");
        this.exists = false;
        return;
      }

      this.U = this.ICspec["U"];
      this.V = this.ICspec["V"];
      this.D = this.ICspec["D"];
    }

    call() {
      if (!this.exists) {
        return;
      }

      const fluid = this.fluid;
      for (let i_y = 0; i_y < fluid.Ny; i_y++) {
        const y = fluid.y_lin[i_y];

        for (let i_x = 0; i_x < fluid.Nx; i_x++) {
          const x = fluid.x_lin[i_x];
          const c = i_y * fluid.Nx + i_x;

          fluid.u[c] = fluid.U[c] = this.U(x, y);
          fluid.v[c] = fluid.V[c] = this.V(x, y);
          fluid.d[c] = this.D(x, y);
        }
      }
    }
  }

  class Fluid {
    constructor(spec, solver) {
      const fluid_spec = spec["fluid"];
      this.name = fluid_spec["name"] !== null
        ? fluid_spec["name"]
        : "unknown_fluid";
      this.solver = solver;

      // Domain
      this.x_max = spec["domain"]["width"];
      this.y_max = spec["domain"]["height"];
      this.base_size = spec["domain"]["base_size"];

      this.Nx = Math.trunc(this.x_max / this.base_size);
      this.Ny = Math.trunc(this.y_max / this.base_size);
      this.N = this.Nx * this.Ny;

      this.dx = this.x_max / this.Nx;
      this.dy = this.y_max / this.Ny;

      this.x_lin = new Float64Array(this.Nx);
      for (let i_x = 0; i_x < this.Nx; i_x++) {
        this.x_lin[i_x] = i_x * this.dx;
      }
      this.y_lin = new Float64Array(this.Ny);
      for (let i_y = 0; i_y < this.Ny; i_y++) {
        this.y_lin[i_y] = i_y * this.dy;
      }

      // Wrapped neighbour lookups, standing in for numpy's shifted slices:
      // i_prev[i_x] is the column left of i_x, row_next[i_y] the flat offset
      // of the row below i_y, and so on
      this.i_prev = new Int32Array(this.Nx);
      this.i_next = new Int32Array(this.Nx);
      for (let i_x = 0; i_x < this.Nx; i_x++) {
        this.i_prev[i_x] = (i_x + this.Nx - 1) % this.Nx;
        this.i_next[i_x] = (i_x + 1) % this.Nx;
      }

      this.row_prev = new Int32Array(this.Ny);
      this.row_next = new Int32Array(this.Ny);
      for (let i_y = 0; i_y < this.Ny; i_y++) {
        this.row_prev[i_y] = ((i_y + this.Ny - 1) % this.Ny) * this.Nx;
        this.row_next[i_y] = ((i_y + 1) % this.Ny) * this.Nx;
      }

      // Initial conditions
      this.u = new Float32Array(this.N);
      this.U = new Float32Array(this.N);
      this.u_tmp = new Float32Array(this.N);

      this.v = new Float32Array(this.N);
      this.V = new Float32Array(this.N);
      this.v_tmp = new Float32Array(this.N);

      this.p = new Float32Array(this.N);
      this.p_tmp = new Float32Array(this.N);

      this.d = new Float32Array(this.N);
      this.d_tmp = new Float32Array(this.N);

      this.div_v = new Float32Array(this.N);
      this.jacobi_tmp = new Float32Array(this.N);

      this.ICs = new InitialCondition(spec, this);
      this.ICs.call();

      // Properties
      this.nu = fluid_spec["viscosity"];
      this.smoke_nu = fluid_spec["smoke_viscosity"];
      this.smoke_fade = fluid_spec["smoke_fade"];

      this.diffuse = null;
      this.set_diffusion_solver();

      this.advect = null;
      this.set_advection_solver();

      this.div = null;
      this.set_div_function();
    }

    set_diffusion_solver() {
      const solver = this.solver;

      if (solver.solver_type === "ExplicitEuler" && solver.dx_is_dy) {
        this.diffuse = (D, D_new, nu = this.nu) => Solver.diffuseEE_dx_is_dy(
          this, D, D_new, nu, this.dx, solver.dt
        );
      } else if (solver.solver_type === "ImplicitEuler" && solver.dx_is_dy) {
        this.diffuse = (D, D_new, nu = this.nu) => Solver.diffuseIE_dx_is_dy(
          this, D, D_new, nu, this.dx, solver.dt, solver.nit
        );
      } else {
        console.warn(`Solver '${solver.solver_type}' not recognised`);
      }
    }

    set_advection_solver() {
      this.advect = (D, Dff) => Solver.advect(
        this, D, Dff, this.u, this.v, this.dx, this.dy, this.solver.dt
      );
    }

    set_div_function() {
      if (this.solver.dx_is_dy) {
        this.div = (u, v, out) => Solver.div_dx_is_dy(this, u, v, out, this.dx);
      } else {
        this.div = (u, v, out) => Solver.div_dx_not_dy(
          this, u, v, out, this.dx, this.dy
        );
      }
    }

    diffuse_velocity() {
      this.diffuse(this.u, this.u_tmp);
      let swap = this.u; this.u = this.u_tmp; this.u_tmp = swap;

      this.diffuse(this.v, this.v_tmp);
      swap = this.v; this.v = this.v_tmp; this.v_tmp = swap;
    }

    enforce_continuity() {
      Solver.extract_divfree(
        this, this.dx, this.dy, this.solver.nit, this.div
      );
    }

    advect_velocity() {
      // Both components are traced back along the same velocity field, so u is
      // not replaced until v has been advected too (the u_tmp in the Python)
      this.advect(this.u, this.u_tmp);
      this.advect(this.v, this.v_tmp);

      let swap = this.u; this.u = this.u_tmp; this.u_tmp = swap;
      swap = this.v; this.v = this.v_tmp; this.v_tmp = swap;
    }

    diffuse_smoke() {
      this.diffuse(this.d, this.d_tmp, this.smoke_nu);
      const swap = this.d; this.d = this.d_tmp; this.d_tmp = swap;
    }

    advect_smoke() {
      this.advect(this.d, this.d_tmp);
      const swap = this.d; this.d = this.d_tmp; this.d_tmp = swap;
    }

    fade_smoke() {
      if (this.smoke_fade === 1) {
        return;
      }
      for (let c = 0; c < this.N; c++) {
        this.d[c] *= this.smoke_fade;
      }
    }
  }

  /* ---------------------------------------------------------------------- */

  class Solver {
    constructor() {
      this.name = spec["name"];

      this.solver_type = spec["scheme"]["name"];
      this.dx_is_dy = spec["scheme"]["dx==dy"];
      this.nit = spec["scheme"]["nit"];
      this.dt = spec["time"]["dt"];
      this.t_max = spec["time"]["t_max"];
      this.t = 0;
      this.fluid = new Fluid(spec, this);

      this.display = new Display(spec, this);
      this.gui = new GUI(spec, this);
      this.mainloop = new Mainloop(this);
    }

    run() {
      this.mainloop.call();
    }

    solve() {
      if (this.t > this.t_max) {
        return 1;
      }

      this.fluid.diffuse_velocity();
      this.fluid.enforce_continuity();
      this.fluid.advect_velocity();
      this.fluid.enforce_continuity();

      this.fluid.diffuse_smoke();
      this.fluid.advect_smoke();
      this.fluid.fade_smoke();
      this.t += this.dt;

      return 0;
    }

    /*
     * Diffuse the scalar field D with the Implicit Euler scheme
     * Assumption: dx = dy
     *
     * Each sweep reads the previous iterate whole, as the numpy slicing does,
     * so the two buffers ping-pong. Starting on the buffer matching the parity
     * of nit leaves the answer in D_new without a final copy.
     */
    static diffuseIE_dx_is_dy(fluid, D, D_new, nu, dx, dt, nit) {
      const { Nx, Ny, i_prev, i_next, row_prev, row_next } = fluid;
      const k = 4 * nu * dt / (dx * dx);
      const scale = 1 / (1 + k);

      let src = nit % 2 === 0 ? D_new : fluid.jacobi_tmp;
      let dst = nit % 2 === 0 ? fluid.jacobi_tmp : D_new;
      src.set(D);

      // iteratively progress D to satisfy the equation
      for (let it = 0; it < nit; it++) {
        for (let i_y = 0; i_y < Ny; i_y++) {
          const row = i_y * Nx;
          const up = row_prev[i_y];
          const down = row_next[i_y];

          for (let i_x = 0; i_x < Nx; i_x++) {
            const c = row + i_x;
            dst[c] = (D[c] + 0.25 * k * (src[down + i_x] + src[up + i_x]
              + src[row + i_next[i_x]] + src[row + i_prev[i_x]])) * scale;
          }
        }

        const swap = src; src = dst; dst = swap;
      }

      return D_new;
    }

    /*
     * Diffuse the scalar field D with the Explicit Euler scheme
     * Assumption: dx = dy
     */
    static diffuseEE_dx_is_dy(fluid, D, D_new, nu, dx, dt) {
      const { Nx, Ny, i_prev, i_next, row_prev, row_next } = fluid;
      const k = 4 * nu * dt / (dx * dx);

      for (let i_y = 0; i_y < Ny; i_y++) {
        const row = i_y * Nx;
        const up = row_prev[i_y];
        const down = row_next[i_y];

        for (let i_x = 0; i_x < Nx; i_x++) {
          const c = row + i_x;
          const M = (D[down + i_x] + D[up + i_x] + D[row + i_next[i_x]]
            + D[row + i_prev[i_x]]) / 4;
          D_new[c] = D[c] * (1 - k) + k * M;
        }
      }

      return D_new;
    }

    /*
     * Advect scalar field D in accordance with the velocity field (u, v)
     */
    static advect(fluid, D, Dff, u, v, dx, dy, dt) {
      const { Nx, Ny, i_next, row_next } = fluid;

      for (let i_y = 0; i_y < Ny; i_y++) {
        const row = i_y * Nx;

        for (let i_x = 0; i_x < Nx; i_x++) {
          const c = row + i_x;

          // IX_prev, IY_prev are the (index) coordinates where we are
          // advecting D from, wrapped back into the domain
          let IX_prev = (i_x - u[c] * dt / dx) % Nx;
          if (IX_prev < 0) IX_prev += Nx;
          if (IX_prev >= Nx) IX_prev -= Nx;   // guards the rounding case -0

          let IY_prev = (i_y - v[c] * dt / dy) % Ny;
          if (IY_prev < 0) IY_prev += Ny;
          if (IY_prev >= Ny) IY_prev -= Ny;

          const x0 = IX_prev | 0;
          const y0 = IY_prev | 0;
          const x1 = i_next[x0];
          const frac_x = IX_prev - x0;
          const frac_y = IY_prev - y0;

          const r0 = y0 * Nx;
          const r1 = row_next[y0];

          const D0f = (1 - frac_x) * D[r0 + x0] + frac_x * D[r0 + x1];
          const D1f = (1 - frac_x) * D[r1 + x0] + frac_x * D[r1 + x1];
          Dff[c] = (1 - frac_y) * D0f + frac_y * D1f;
        }
      }

      return Dff;
    }

    static div_dx_not_dy(fluid, v_x, v_y, out, dx, dy) {
      const { Nx, Ny, i_prev, i_next, row_prev, row_next } = fluid;

      for (let i_y = 0; i_y < Ny; i_y++) {
        const row = i_y * Nx;
        const up = row_prev[i_y];
        const down = row_next[i_y];

        for (let i_x = 0; i_x < Nx; i_x++) {
          out[row + i_x] = (
            (v_x[row + i_next[i_x]] - v_x[row + i_prev[i_x]]) / dx
            + (v_y[down + i_x] - v_y[up + i_x]) / dy) / 2;
        }
      }

      return out;
    }

    static div_dx_is_dy(fluid, v_x, v_y, out, dx) {
      const { Nx, Ny, i_prev, i_next, row_prev, row_next } = fluid;

      for (let i_y = 0; i_y < Ny; i_y++) {
        const row = i_y * Nx;
        const up = row_prev[i_y];
        const down = row_next[i_y];

        for (let i_x = 0; i_x < Nx; i_x++) {
          out[row + i_x] = ((v_x[row + i_next[i_x]] - v_x[row + i_prev[i_x]])
            + (v_y[down + i_x] - v_y[up + i_x])) / (2 * dx);
        }
      }

      return out;
    }

    /* Solve for the pressure, then subtract its gradient from (u, v) */
    static extract_divfree(fluid, dx, dy, nit, div) {
      const { Nx, Ny, i_prev, i_next, row_prev, row_next, u, v } = fluid;
      const div_v = div(u, v, fluid.div_v);

      const dx2 = dx * dx;
      const dy2 = dy * dy;
      const scale = 1 / (2 * (dy2 + dx2));

      // f carries over from the previous step, warm-starting the solve. As in
      // the diffusion, the parity of nit picks the buffer to start on
      let src = fluid.p;
      let dst = fluid.p_tmp;
      if (nit % 2 !== 0) {
        dst.set(src);
        src = fluid.p_tmp;
        dst = fluid.p;
      }

      for (let it = 0; it < nit; it++) {
        for (let i_y = 0; i_y < Ny; i_y++) {
          const row = i_y * Nx;
          const up = row_prev[i_y];
          const down = row_next[i_y];

          for (let i_x = 0; i_x < Nx; i_x++) {
            const c = row + i_x;
            dst[c] = ((src[row + i_next[i_x]] + src[row + i_prev[i_x]]) * dy2
              + (src[down + i_x] + src[up + i_x]) * dx2
              - dx2 * dy2 * div_v[c]) * scale;
          }
        }

        const swap = src; src = dst; dst = swap;
      }

      const f = src;   // === fluid.p

      for (let i_y = 0; i_y < Ny; i_y++) {
        const row = i_y * Nx;
        const up = row_prev[i_y];
        const down = row_next[i_y];

        for (let i_x = 0; i_x < Nx; i_x++) {
          const c = row + i_x;
          const u_cf = (f[row + i_next[i_x]] - f[row + i_prev[i_x]]) / (2 * dx);
          const v_cf = (f[down + i_x] - f[up + i_x]) / (2 * dy);
          u[c] -= u_cf;
          v[c] -= v_cf;
        }
      }
    }
  }

  /* ---------------------------------------------------------------------- */

  class Display {
    constructor(spec, solver) {
      this.Nx = Math.trunc(
        spec["domain"]["width"] / spec["domain"]["base_size"]);
      this.Ny = Math.trunc(
        spec["domain"]["height"] / spec["domain"]["base_size"]);

      this.canvas = document.getElementById("fluid-sim");
      this.ctx = this.canvas.getContext("2d");

      // The fluid is drawn at grid resolution then scaled up - the equivalent
      // of pygame's make_surface followed by transform.scale
      this.fluid_surf = document.createElement("canvas");
      this.fluid_surf.width = this.Nx;
      this.fluid_surf.height = this.Ny;
      this.fluid_ctx = this.fluid_surf.getContext("2d");
      this.pxarray = this.fluid_ctx.createImageData(this.Nx, this.Ny);

      this.sf = 1;
      this.blit_offset = [0, 0];
      this.domain_dims = [this.Nx, this.Ny];
      this.dims = [this.Nx, this.Ny];
      this.dpr = 1;

      this.show_smoke = spec["display"]["show_smoke"];
      this.background_colour = [0, 0, 0];
      this.fluid_colour = [0, 0, 0];

      this.solver = solver;
      this.fluid = solver.fluid;

      this.update_transformation();
    }

    call() {
      this.update_pxarray();
      this.blit_pxarray();
    }

    update_transformation() {
      const rect = this.canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) {
        return;   // not laid out yet; the ResizeObserver will call again
      }
      this.dims = [rect.width, rect.height];

      const AR_px = this.Nx / this.Ny;
      const AR_wind = this.dims[0] / this.dims[1];

      this.sf = AR_px > AR_wind
        ? this.dims[0] / this.Nx
        : this.dims[1] / this.Ny;

      this.domain_dims = [this.Nx * this.sf, this.Ny * this.sf];
      this.blit_offset = [
        (this.dims[0] - this.domain_dims[0]) / 2,
        (this.dims[1] - this.domain_dims[1]) / 2
      ];

      // Back the canvas with real device pixels so the upscale stays crisp
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.canvas.width = Math.max(1, Math.round(this.dims[0] * this.dpr));
      this.canvas.height = Math.max(1, Math.round(this.dims[1] * this.dpr));
      this.ctx.imageSmoothingEnabled = true;
    }

    update_pxarray() {
      const px = this.pxarray.data;
      const d = this.fluid.d;
      const [r, g, b] = this.fluid_colour;
      const show_smoke = this.show_smoke;

      // px is a Uint8ClampedArray, so writing to it clips to 0-255 the way
      // update_pxarray's np.clip does
      for (let c = 0, o = 0; c < d.length; c++, o += 4) {
        const s = show_smoke ? d[c] : 0;
        px[o] = r + s;
        px[o + 1] = g + s;
        px[o + 2] = b + s;
        px[o + 3] = 255;
      }
    }

    blit_pxarray() {
      this.fluid_ctx.putImageData(this.pxarray, 0, 0);

      this.ctx.fillStyle = `rgb(${this.background_colour.join(",")})`;
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      this.ctx.drawImage(
        this.fluid_surf,
        this.blit_offset[0] * this.dpr, this.blit_offset[1] * this.dpr,
        this.domain_dims[0] * this.dpr, this.domain_dims[1] * this.dpr
      );
    }
  }

  /* ---------------------------------------------------------------------- */

  class Mouse {
    constructor() {
      this.pos = null;
      this.pos_prev = null;
      this.pos_stack = [];
      this.l_press = 0;
      this.mid_press = 0;
      this.r_press = 0;

      this.state = 0;
      //  0: not pressed
      // -1: just unpressed
      //  1: held down
      //  2: just pressed

      // Filled by the pointer listeners and drained once per frame, the way
      // pygame's event queue is
      this.events = [];
      this.pressed = [0, 0, 0];
      this.live_pos = null;
    }

    init() {
      this.get_state();
      this.get_pos_stack();
      this.get_pos();
    }

    get_state() {
      this.l_press = this.pressed[0];
      this.mid_press = this.pressed[1];
      this.r_press = this.pressed[2];

      if (this.l_press) {
        if (this.state === 2) {
          this.state = 1;
        } else if (this.state !== 1) {
          this.state = 2;
        }
      } else {
        if (this.state === -1) {
          this.state = 0;
        } else if (this.state !== 0) {
          this.state = -1;
        }
      }
    }

    get_pos() {
      if (this.pos !== null) {
        this.pos_prev = this.pos.slice();
      }

      this.pos = this.live_pos !== null ? this.live_pos.slice() : [0, 0];

      if (this.pos_prev === null) {
        this.pos_prev = this.pos.slice();
      }
    }

    get_pos_stack() {
      this.pos_stack = this.events;
      this.events = [];

      if (!this.pos_stack.length && this.live_pos !== null) {
        // if not moving
        this.pos_stack.push(this.live_pos.slice());
      }
    }
  }

  class GUI {
    constructor(spec, solver) {
      this.mouse = new Mouse();
      this.solver = solver;
      this.fluid = solver.fluid;
      this.display = solver.display;

      this.smoke_strength = spec["gui"]["smoke_strength"];
      this.brush_size = spec["gui"]["brush_size"];
      this.origin_brush = null;
      this.set_origin_brush();

      this.bind_events();
    }

    call() {
      this.mouse.init();
      this.fluid_interaction();
    }

    set_origin_brush() {
      const rad = Math.max(1, Math.trunc(this.brush_size / this.display.sf));
      const rows = [];
      const cols = [];

      for (let i = -rad; i < rad; i++) {
        for (let j = -rad; j < rad; j++) {
          const X = j + 0.5;
          const Y = i + 0.5;
          if (X * X + Y * Y <= rad * rad) {
            rows.push(i);
            cols.push(j);
          }
        }
      }

      this.origin_brush = [Int32Array.from(rows), Int32Array.from(cols)];
    }

    /* The brush wraps with the domain, so it stays whole across an edge */
    move_brush(pos) {
      const [rows, cols] = this.origin_brush;
      const { Nx, Ny } = this.fluid;
      const brush = new Int32Array(rows.length);

      for (let n = 0; n < rows.length; n++) {
        const i_y = (((rows[n] + pos[0]) % Ny) + Ny) % Ny;
        const i_x = (((cols[n] + pos[1]) % Nx) + Nx) % Nx;
        brush[n] = i_y * Nx + i_x;
      }

      return brush;
    }

    fluid_interaction() {
      const mouse = this.mouse;
      if (!(mouse.l_press || mouse.r_press)) {
        return;
      }

      for (const mouse_pos of mouse.pos_stack) {
        // the [::-1] in the Python: screen (x, y) becomes index (i_y, i_x)
        const mouse_index = [
          Math.trunc(
            (mouse_pos[1] - this.display.blit_offset[1]) / this.display.sf),
          Math.trunc(
            (mouse_pos[0] - this.display.blit_offset[0]) / this.display.sf)
        ];

        if (0 <= mouse_index[1] && mouse_index[1] < this.fluid.Nx
          && 0 <= mouse_index[0] && mouse_index[0] < this.fluid.Ny) {
          const delta_pos = [
            mouse.pos[0] - mouse.pos_prev[0],
            mouse.pos[1] - mouse.pos_prev[1]
          ];
          const brush_pos = this.move_brush(mouse_index);

          this.add_smoke(brush_pos);
          this.push_fluid(brush_pos, delta_pos);
        }
      }
    }

    add_smoke(brush_pos) {
      // dividing by pos_stack ensures constant smoke addition per
      // time step
      // dividing by base_size^2 ensures constant smoke addition
      // per unit area
      if (!this.mouse.l_press) {
        return;
      }

      const amount = this.smoke_strength
        / (this.mouse.pos_stack.length * this.fluid.base_size ** 2);

      for (let n = 0; n < brush_pos.length; n++) {
        this.fluid.d[brush_pos[n]] += amount;
      }
    }

    push_fluid(brush_pos, delta_pos) {
      // delta_pos is in screen pixels; dividing by sf puts it in cells first,
      // so the fluid follows the cursor rather than overshooting by the zoom
      const u = delta_pos[0] / this.display.sf * this.fluid.dx / this.solver.dt;
      const v = delta_pos[1] / this.display.sf * this.fluid.dy / this.solver.dt;

      for (let n = 0; n < brush_pos.length; n++) {
        this.fluid.u[brush_pos[n]] = u;
        this.fluid.v[brush_pos[n]] = v;
      }
    }

    bind_events() {
      const canvas = this.display.canvas;
      const mouse = this.mouse;

      const to_pos = (event) => {
        const rect = canvas.getBoundingClientRect();
        return [event.clientX - rect.left, event.clientY - rect.top];
      };

      canvas.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        canvas.setPointerCapture(event.pointerId);

        mouse.pressed[event.button === 2 ? 2 : event.button === 1 ? 1 : 0] = 1;
        mouse.live_pos = to_pos(event);
        // Drop the stale position, so the press itself is not read as a drag
        mouse.pos = null;
        mouse.pos_prev = null;

        const hint = document.getElementById("fluid-sim-hint");
        if (hint) {
          hint.classList.add("is-hidden");
        }
      });

      canvas.addEventListener("pointermove", (event) => {
        mouse.live_pos = to_pos(event);
        if (!(mouse.pressed[0] || mouse.pressed[2])) {
          return;
        }
        event.preventDefault();

        // Coalesced events recover the positions the browser batched into this
        // one - the closest thing to pygame's queue of MOUSEMOTION events
        const events = event.getCoalescedEvents
          ? event.getCoalescedEvents()
          : [event];
        for (const e of events) {
          mouse.events.push(to_pos(e));
        }
      });

      const release = (event) => {
        if (canvas.hasPointerCapture(event.pointerId)) {
          canvas.releasePointerCapture(event.pointerId);
        }
        mouse.pressed = [0, 0, 0];
      };

      canvas.addEventListener("pointerup", release);
      canvas.addEventListener("pointercancel", release);
      canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    }
  }

  /* ---------------------------------------------------------------------- */

  class Mainloop {
    constructor(solver) {
      this.solver = solver;
      this.display = solver.display;
      this.gui = solver.gui;

      // Only step while the canvas is on screen
      this.visible = true;
      if (window.IntersectionObserver) {
        new IntersectionObserver((entries) => {
          this.visible = entries[0].isIntersecting;
        }, { threshold: 0 }).observe(this.display.canvas);
      }

      // Fires once on setup as well as on every resize, which also covers the
      // case of the canvas not having been laid out yet
      if (window.ResizeObserver) {
        new ResizeObserver(() => {
          this.display.update_transformation();
          this.gui.set_origin_brush();
        }).observe(this.display.canvas);
      } else {
        window.addEventListener("resize", () => {
          this.display.update_transformation();
          this.gui.set_origin_brush();
        });
      }
    }

    init() {
      this.gui.call();
      return 0;
    }

    call() {
      const frame = () => {
        if (this.visible) {
          if (!this.init() && !this.solver.solve()) {
            this.display.call();
          }
        }
        window.requestAnimationFrame(frame);
      };

      window.requestAnimationFrame(frame);
    }
  }

  /* ---------------------------------------------------------------------- */

  function main() {
    if (!document.getElementById("fluid-sim")) {
      return;
    }

    const solver = new Solver();

    const reset = document.getElementById("fluid-sim-reset");
    if (reset) {
      reset.addEventListener("click", () => {
        solver.fluid.p.fill(0);
        solver.fluid.ICs.call();
        solver.t = 0;

        const hint = document.getElementById("fluid-sim-hint");
        if (hint) {
          hint.classList.remove("is-hidden");
        }
      });
    }

    solver.run();
  }

  main();
})();
