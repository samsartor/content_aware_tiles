var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => {
  __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
  return value;
};
const tiling_shader = "vec4 blend(vec4 b, vec4 a) {\n  return vec4(a.rgb * a.a + b.rgb * b.a * (1.0f - a.a), a.a + b.a * (1.0f - a.a));\n}\n\nuint lowbias32(uint x) {\n  x ^= x >> 17;\n  x *= 0xed5ad4bbU;\n  x ^= x >> 11;\n  x *= 0xac4c1b51U;\n  x ^= x >> 15;\n  x *= 0x31848babU;\n  x ^= x >> 14;\n  return x;\n}\n\nint tileEdgeRandom(int x, int y, uint div, uint rem) {\n  uint i = lowbias32(lowbias32(uint(x)) + uint(y));\n  return int((i / div) % rem);\n}\n\nstruct TileColors {\n  int l;\n  int t;\n  int r;\n  int b;\n};\n\nTileColors interiorTileColors(vec2 i, uint colors) {\n  int l = tileEdgeRandom(int(i.x), int(i.y), 1U, colors);\n  int t = tileEdgeRandom(int(i.x), int(i.y), colors, colors);\n  int r = tileEdgeRandom(int(i.x) + 1, int(i.y), 1U, colors);\n  int b = tileEdgeRandom(int(i.x), int(i.y) + 1, colors, colors);\n  return TileColors(l, t, r, b);\n}\n\nTileColors crossTileColors(vec2 i, uint colors) {\n  int l = tileEdgeRandom(int(i.x) - 1, int(i.y), colors, colors);\n  int t = tileEdgeRandom(int(i.x), int(i.y) - 1, 1U, colors);\n  int r = tileEdgeRandom(int(i.x), int(i.y), colors, colors);\n  int b = tileEdgeRandom(int(i.x), int(i.y), 1U, colors);\n  return TileColors(l, t, r, b);\n}\n\nstruct Details {\n  ivec2 selfTiles;\n  int selfOffset;\n  float wireframeAlpha;\n  float wireframeThickness;\n  bool flipY;\n};\n\nvec4 tileWireframe(TileColors c, vec2 f, int mode, Details details) {\n  vec3 colors[6] = vec3[](vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 1.0f, 1.0f), vec3(1.0f, 0.0f, 1.0f), vec3(1.0f, 1.0f, 0.0f));\n  float thick = details.wireframeThickness;\n  float thin = 1.0 - thick;\n  if (mode == 0 || mode == 1) {\n    float r = max(abs(f.x), abs(f.y));\n    if(r > thin && r < 1.0 + thick) {\n      return vec4(1.0f, 1.0f, 1.0f, details.wireframeAlpha);\n    }\n  } else if (mode == 2) {\n    float r = max(abs(f.x), abs(f.y));\n    if(f.x <= f.y && f.x < -f.y && r > thin && r < 1.0 + thick) {\n        return vec4(colors[c.l], details.wireframeAlpha);\n    }\n    if(f.x >= f.y && f.x > -f.y && r > thin && r < 1.0 + thick) {\n        return vec4(colors[c.r], details.wireframeAlpha);\n    }\n    if(f.x > f.y && f.x <= -f.y && r > thin && r < 1.0 + thick) {\n        return vec4(colors[c.t], details.wireframeAlpha);\n    }\n    if(f.x < f.y && f.x >= -f.y && r > thin && r < 1.0 + thick) {\n        return vec4(colors[c.b], details.wireframeAlpha);\n    }\n  } else if (mode == 3) {\n    float tdist = length(f - vec2( 0.0f, -1.0f));\n    float bdist = length(f - vec2( 0.0f,  1.0f));\n    float ldist = length(f - vec2(-1.0f, 0.0f));\n    float rdist = length(f - vec2( 1.0f,  0.0f));\n\n    if (tdist < thick && tdist < bdist && tdist < ldist && tdist < rdist) {\n      return vec4(colors[c.t], details.wireframeAlpha);\n    }\n    if (ldist < thick && ldist < bdist && ldist < tdist && ldist < rdist) {\n      return vec4(colors[c.l], details.wireframeAlpha);\n    }\n    if (bdist < thick && bdist < ldist && bdist < tdist && bdist < rdist) {\n      return vec4(colors[c.b], details.wireframeAlpha);\n    }\n    if (rdist < thick && rdist < ldist && rdist < tdist && rdist < bdist) {\n      return vec4(colors[c.r], details.wireframeAlpha);\n    }\n  }\n  return vec4(0.0f, 0.0f, 0.0f, 0.0f);\n}\n\nconst int INVERSE_PACKING[162] = int[](0, 60, 40, 57, 27, 16, 41, 11, 72, 50, 10, 70, 26, 67, 37, 1, 51, 21, 20, 30, 80, 77, 6, 47, 61, 71, 31, 49, 19, 8, 17, 68, 48, 65, 44, 24, 9, 59, 29, 58, 18, 78, 34, 75, 45, 69, 79, 39, 28, 38, 7, 4, 14, 55, 25, 76, 56, 33, 3, 64, 73, 52, 32, 66, 35, 5, 74, 43, 13, 42, 2, 62, 36, 46, 15, 53, 54, 23, 12, 22, 63, 131, 90, 147, 101, 150, 117, 81, 130, 106, 107, 139, 155, 158, 109, 134, 138, 98, 114, 82, 115, 123, 142, 85, 93, 122, 146, 154, 91, 140, 116, 141, 100, 157, 111, 160, 127, 148, 99, 124, 108, 149, 84, 87, 119, 135, 132, 156, 83, 92, 125, 133, 152, 95, 103, 151, 110, 86, 161, 120, 96, 121, 89, 137, 118, 159, 94, 128, 88, 104, 97, 129, 145, 102, 126, 143, 112, 136, 144, 153, 105, 113);\n\nint tileIndex(TileColors neighbors, int offset) {\n  int tile = neighbors.l + neighbors.t * 3 + neighbors.r * 9 + neighbors.b * 27 + offset;\n  return int(INVERSE_PACKING[tile]);\n}\n\nvec4 textureTiling(sampler2D image, vec2 uv, int mode, Details details) {\n  vec2 f = fract(uv);\n  vec2 i = uv - f;\n  f = f * 2.0f - 1.0f;\n\n  TileColors neighbors;\n  vec2 tile_uv;\n  vec2 par_uv = uv;\n\n  vec2 selfTilesF = vec2(float(details.selfTiles.x), float(details.selfTiles.y));\n\n  if (mode == 0 || mode == 1) {\n    // Stochastic Self-tiling Tiles\n    neighbors = TileColors(0, 0, 0, 0);\n    int index;\n    if (mode == 1) {\n      index = tileEdgeRandom(int(i.x), int(i.y), 1U, uint(details.selfTiles.x * details.selfTiles.y));\n    } else {\n      index = 0;\n    }\n    index += details.selfOffset;\n    tile_uv = f * 0.5f + 0.5f;\n    tile_uv += vec2(float(index % details.selfTiles.x), float(index / details.selfTiles.x));\n    tile_uv /= selfTilesF;\n    par_uv /= selfTilesF;\n  } else {\n    // Wang Tiles (or Interior Tiles)\n    neighbors = interiorTileColors(i, 3U);\n    int index = tileIndex(neighbors, 0);\n\n    tile_uv = f * 0.5f + 0.5f;\n\n    vec2 cross_uv = uv + 0.5;\n    vec2 cross_f = fract(cross_uv);\n    vec2 cross_i = cross_uv - cross_f;\n    cross_f = cross_f * 2.0 - 1.0;\n    if(mode == 3 && abs(cross_f.x) + abs(cross_f.y) < abs(f.x) + abs(f.y)) {\n      // Cross Tiles\n      f = cross_f;\n      tile_uv = f * 0.5f;\n      neighbors = crossTileColors(cross_i, 3U);\n      index = tileIndex(neighbors, 81);\n    }\n\n    tile_uv += vec2(float(index % 9), float(index / 9));\n    tile_uv /= 9.0f;\n    par_uv /= 9.0f;\n  }\n\n  if (details.flipY) {\n    tile_uv.y = 1.0 - tile_uv.y;\n  }\n\n  vec4 wireframe_color = tileWireframe(neighbors, f, mode, details);\n  vec4 texture_color = textureGrad(image, tile_uv, dFdx(par_uv), dFdy(par_uv));\n  return blend(texture_color, wireframe_color);\n}\n";
const drag_arrow = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<svg\n   viewBox="0 0 64 64"\n   version="1.1"\n   xmlns="http://www.w3.org/2000/svg"\n   xmlns:svg="http://www.w3.org/2000/svg"\n   id="dragarrow">\n   <path d="M 31.999487,0 21.325482,14.023076 h 5.719279 V 27.045783 H 14.023077 V 21.325482 L 0,32.000513 14.023077,42.674519 v -5.719276 h 13.02271 V 49.976667 H 21.325482 L 32.000512,64 42.674519,49.976667 h -5.719277 v -13.02245 h 13.021425 v 5.720302 L 64,31.999487 49.976667,21.325482 v 5.71928 H 36.954217 V 14.023076 h 5.720302 z" />\n</svg>\n';
function lerp(t, a, b) {
  return (1 - t) * a + t * b;
}
class TilingCanvas extends HTMLElement {
  constructor() {
    super();
    __publicField(this, "canvas");
    __publicField(this, "gl");
    __publicField(this, "uniforms", /* @__PURE__ */ new Map());
    __publicField(this, "resizeObserver", null);
    __publicField(this, "loaded", false);
    __publicField(this, "isDragging", false);
    __publicField(this, "mode", 3);
    __publicField(this, "wireframe", 0);
    __publicField(this, "startX", 0);
    __publicField(this, "startY", 0);
    __publicField(this, "startD", null);
    __publicField(this, "selfTilesX", 1);
    __publicField(this, "selfTilesY", 1);
    __publicField(this, "minX", -15);
    __publicField(this, "minY", -15);
    __publicField(this, "maxX", 15);
    __publicField(this, "maxY", 15);
    this.attachShadow({ mode: "open" });
    this.canvas = document.createElement("canvas");
    let hint;
    if (navigator.maxTouchPoints > 1) {
      hint = "drag to pan, pinch to zoom";
    } else {
      hint = "drag to pan, ctrl+scroll to zoom";
    }
    this.shadowRoot.innerHTML += `
      <style>
      :host {
        position: relative;
        overflow: hidden;
        display: block;
        cursor: pointer;
      }

      svg {
        fill: white;
        stroke: black;
        stroke-width: 2px;
        position: absolute;
        right: 8px;
        bottom: 8px;
        width: 64px;
      }

      #hint {
        display: none;
        position: absolute;
        color: black;
        background: white;
        border-radius: 2px;
        right: 80px;
        bottom: 8px;
        padding: 4px;
        border: solid black 2px;
      }

      svg:hover + #hint {
          display: block;
      }
      </style>
      ${drag_arrow}
      <span id="hint">
        ${hint}
      </span>
    `;
    this.shadowRoot.appendChild(this.canvas);
    this.gl = this.canvas.getContext("webgl2");
    this.shadowRoot.getElementById("dragarrow").addEventListener("click", () => {
      this.setAttribute("wireframe", "" + (this.wireframe > 0 ? 0 : 0.15));
    });
  }
  connectedCallback() {
    this.setupEvents();
    this.applyAttributes();
  }
  disconnectedCallback() {
    this.resizeObserver.disconnect();
  }
  attributeChangedCallback(name, oldValue, newValue) {
    this.applyAttributes();
    if (name == "src") {
      this.loadImage(this.getAttribute("src"));
    } else {
      this.render();
    }
  }
  setupEvents() {
    this.addEventListener("mousedown", (e) => this.onDragStart(e));
    this.addEventListener("touchstart", (e) => this.onDragStart(e));
    window.addEventListener("mouseup", () => this.onDragEnd());
    window.addEventListener("touchend", () => this.onDragEnd());
    window.addEventListener("mousemove", (e) => this.onDragMove(e));
    window.addEventListener("touchmove", (e) => this.onDragMove(e), { passive: false });
    this.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    this.resizeObserver = new ResizeObserver(() => this.resizeCanvas());
    this.resizeObserver.observe(this);
  }
  resizeCanvas() {
    const rect = this.getBoundingClientRect();
    if (this.canvas.width != rect.width || this.canvas.height != rect.height) {
      this.canvas.width = Math.ceil(rect.width);
      this.canvas.height = Math.ceil(rect.height);
      const aspect = this.canvas.height / this.canvas.width;
      const radius = (this.maxX - this.minX) * aspect / 2;
      const center = (this.maxY + this.minY) / 2;
      this.minY = center - radius;
      this.maxY = center + radius;
      if (this.gl) {
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.render();
      }
    }
  }
  applyAttributes() {
    this.wireframe = parseFloat(this.getAttribute("wireframe") || "0");
    if (!isFinite(this.wireframe)) {
      this.wireframe = 0;
    }
    this.mode = {
      "single": 0,
      "self": 1,
      "wang": 2,
      "dual": 3
    }[this.getAttribute("mode") || "dual"] || 3;
    const selfTiles = (this.getAttribute("selfTiles") || "1,1").split(",");
    this.selfTilesX = parseInt(selfTiles[0]);
    this.selfTilesY = parseInt(selfTiles[1]);
  }
  onDragStart(e) {
    this.isDragging = true;
    this.startX = this.getEventX(e);
    this.startY = this.getEventY(e);
  }
  onDragEnd() {
    this.isDragging = false;
    this.startD = null;
  }
  onDragMove(e) {
    if (this.isDragging) {
      const currentX = this.getEventX(e);
      const currentY = this.getEventY(e);
      const dx = currentX - this.startX;
      const dy = currentY - this.startY;
      this.minX -= dx;
      this.maxX -= dx;
      this.minY -= dy;
      this.maxY -= dy;
      const currentD = this.getEventD(e);
      if (this.startD === null) {
        this.startD = currentD;
      }
      if (currentD !== null && this.startD !== null) {
        const zoomFactor = this.startD / currentD;
        this.minX = (this.minX - currentX) * zoomFactor + currentX;
        this.maxX = (this.maxX - currentX) * zoomFactor + currentX;
        this.minY = (this.minY - currentY) * zoomFactor + currentY;
        this.maxY = (this.maxY - currentY) * zoomFactor + currentY;
      }
      this.render();
      e.preventDefault();
    }
  }
  onWheel(e) {
    if (!e.ctrlKey) {
      return;
    }
    e.preventDefault();
    const mouseX = this.getEventX(e);
    const mouseY = this.getEventY(e);
    const zoomFactor = Math.pow(2, e.deltaY / 500);
    this.minX = (this.minX - mouseX) * zoomFactor + mouseX;
    this.maxX = (this.maxX - mouseX) * zoomFactor + mouseX;
    this.minY = (this.minY - mouseY) * zoomFactor + mouseY;
    this.maxY = (this.maxY - mouseY) * zoomFactor + mouseY;
    this.render();
  }
  getEventX(e, touch = 0) {
    let x;
    if (e instanceof MouseEvent) {
      x = e.clientX;
    } else {
      x = e.touches[touch].clientX;
    }
    const rect = this.canvas.getBoundingClientRect();
    return lerp((x - rect.left) / rect.width, this.minX, this.maxX);
  }
  getEventY(e, touch = 0) {
    let y;
    if (e instanceof MouseEvent) {
      y = e.clientY;
    } else {
      y = e.touches[touch].clientY;
    }
    const rect = this.canvas.getBoundingClientRect();
    return lerp((y - rect.top) / rect.height, this.minY, this.maxY);
  }
  getEventD(e) {
    if (e instanceof MouseEvent || e.touches.length < 2) {
      return null;
    } else {
      const dx = this.getEventX(e, 1) - this.getEventX(e, 0);
      const dy = this.getEventY(e, 1) - this.getEventY(e, 0);
      return Math.sqrt(dx * dx + dy * dy);
    }
  }
  loadImage(src) {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      this.initWebGL(image);
    };
    image.src = src;
  }
  initWebGL(image) {
    const gl = this.gl;
    const vertexShaderSource = `#version 300 es

      in vec2 a_position;
      in vec2 a_texCoord;

      out vec2 v_uv;

      uniform vec2 u_mins;
      uniform vec2 u_maxs;

      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_uv = a_texCoord * (u_maxs - u_mins) + u_mins;
      }
    `;
    const vertexShader = this.createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShaderSource = `#version 300 es

      precision highp float;
      precision highp int;

      in vec2 v_uv;
      out vec4 out_color;

      uniform sampler2D u_image;
      uniform int u_mode;
      uniform ivec2 u_self_tiles;
      uniform float u_wireframe;

      ${tiling_shader}

      void main() {
          out_color = textureTiling(u_image, v_uv, u_mode, Details(u_self_tiles, 0, u_wireframe > 0.0 ? 1.0 : 0.0, u_wireframe, false));
      }
    `;
    const fragmentShader = this.createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
    const program = this.createProgram(gl, vertexShader, fragmentShader);
    const positionLocation = gl.getAttribLocation(program, "a_position");
    const texCoordLocation = gl.getAttribLocation(program, "a_texCoord");
    for (const uniform of ["u_mins", "u_maxs", "u_mode", "u_wireframe", "u_self_tiles"]) {
      this.uniforms.set(uniform, gl.getUniformLocation(program, uniform));
    }
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1,
      1,
      1,
      1,
      -1,
      -1,
      -1,
      -1,
      1,
      1,
      1,
      -1
    ]), gl.STATIC_DRAW);
    const texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      0,
      0,
      1,
      0,
      0,
      1,
      0,
      1,
      1,
      0,
      1,
      1
    ]), gl.STATIC_DRAW);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program);
    gl.enableVertexAttribArray(positionLocation);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(texCoordLocation);
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 0, 0);
    this.loaded = true;
    this.render();
  }
  render() {
    if (!this.loaded) {
      return;
    }
    const gl = this.gl;
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform2f(this.uniforms.get("u_mins"), this.minX, this.minY);
    gl.uniform2f(this.uniforms.get("u_maxs"), this.maxX, this.maxY);
    gl.uniform1i(this.uniforms.get("u_mode"), this.mode);
    gl.uniform2i(this.uniforms.get("u_self_tiles"), this.selfTilesX, this.selfTilesY);
    gl.uniform1f(this.uniforms.get("u_wireframe"), this.wireframe);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
  createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }
  createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(program));
      gl.deleteProgram(program);
      return null;
    }
    return program;
  }
}
__publicField(TilingCanvas, "observedAttributes", ["src", "wireframe"]);
customElements.define("tiling-canvas", TilingCanvas);
