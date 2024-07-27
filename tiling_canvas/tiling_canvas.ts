import tiling_shader from "./tiling_shader.glsl?raw";
import drag_arrow from "./drag_arrow.svg?raw"

function lerp(t, a, b) {
  return (1 - t) * a + t * b;
}

function ilerp(x, a, b) {
  return (x - a) / (b - a);
}

class TilingCanvas extends HTMLElement {
  static observedAttributes = ['src', 'wireframe'];

  canvas: HTMLCanvasElement;
  gl: WebGL2RenderingContext;
  uniforms: Map<string, WebGLUniformLocation> = new Map();
  resizeObserver: ResizeObserver | null = null;
  loaded: boolean = false;

  isDragging: boolean = false;
  mode: number = 3;
  wireframe: number = 0;
  startX: number = 0;
  startY: number = 0;
  startD: number | null = null;
  selfTilesX: number = 1;
  selfTilesY: number = 1;
  minX: number = -15;
  minY: number = -15;
  maxX: number = 15;
  maxY: number = 15;

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this.canvas = document.createElement('canvas');

    let hint;
    if (navigator.maxTouchPoints > 1)  {
      hint = 'drag to pan, pinch to zoom';
    } else {
      hint = 'drag to pan, ctrl+scroll to zoom';
    }

    this.shadowRoot!.innerHTML += `
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

      span {
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

      svg:hover + span {
          display: block;
      }
      </style>
      ${drag_arrow}
      <span>${hint}</span>
    `;
    this.shadowRoot!.appendChild(this.canvas);
    this.gl = this.canvas.getContext('webgl2')!;
  }

  connectedCallback() {
    this.setupEvents();
    this.applyAttributes();
  }

  disconnectedCallback() {
    this.resizeObserver!.disconnect();
  }

  attributeChangedCallback(name: string, oldValue, newValue) {
    this.applyAttributes();
    if (name == 'src') {
      this.loadImage(this.getAttribute('src'));
    } else {
      this.render();
    }
  }

  setupEvents() {
    this.addEventListener('mousedown', (e) => this.onDragStart(e));
    this.addEventListener('touchstart', (e) => this.onDragStart(e));

    window.addEventListener('mouseup', () => this.onDragEnd());
    window.addEventListener('touchend', () => this.onDragEnd());

    window.addEventListener('mousemove', (e) => this.onDragMove(e));
    window.addEventListener('touchmove', (e) => this.onDragMove(e), { passive: false });

    this.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });

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
    this.wireframe = parseFloat(this.getAttribute('wireframe') || '0');
    if (!isFinite(this.wireframe)) {
      this.wireframe = 0.0;
    }

    this.mode = {
      'single': 0,
      'self': 1,
      'wang': 2,
      'dual': 3,
    }[this.getAttribute('mode') || 'dual'] || 3;
    const selfTiles = (this.getAttribute('selfTiles') || '1,1').split(',');
    this.selfTilesX = parseInt(selfTiles[0]);
    this.selfTilesY = parseInt(selfTiles[1]);
  }

  onDragStart(e: MouseEvent | TouchEvent) {
    this.isDragging = true;
    this.startX = this.getEventX(e);
    this.startY = this.getEventY(e);
  }

  onDragEnd() {
    this.isDragging = false;
    this.startD = null;
  }

  onDragMove(e: MouseEvent | TouchEvent) {
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

  onWheel(e: WheelEvent) {
    if (!e.ctrlKey) {
      return;
    }

    e.preventDefault();

    const mouseX = this.getEventX(e);
    const mouseY = this.getEventY(e);

    const zoomFactor = Math.pow(2.0, e.deltaY / 500);

    this.minX = (this.minX - mouseX) * zoomFactor + mouseX;
    this.maxX = (this.maxX - mouseX) * zoomFactor + mouseX;
    this.minY = (this.minY - mouseY) * zoomFactor + mouseY;
    this.maxY = (this.maxY - mouseY) * zoomFactor + mouseY;

    this.render();
  }

  getEventX(e: MouseEvent | TouchEvent, touch = 0) {
    let x: number;
    if (e instanceof MouseEvent) {
      x = e.clientX;
    } else {
      x = e.touches[touch].clientX;
    }
    const rect = this.canvas.getBoundingClientRect();
    return lerp((x - rect.left) / rect.width, this.minX, this.maxX);
  }

  getEventY(e: MouseEvent | TouchEvent, touch = 0) {
    let y: number;
    if (e instanceof MouseEvent) {
      y = e.clientY;
    } else {
      y = e.touches[touch].clientY;
    }
    const rect = this.canvas.getBoundingClientRect();
    return lerp((y - rect.top) / rect.height, this.minY, this.maxY);
  }

  getEventD(e: MouseEvent | TouchEvent): number | null {
    if (e instanceof MouseEvent || e.touches.length < 2) {
      return null;
    } else {
      const dx = this.getEventX(e, 1) - this.getEventX(e, 0);
      const dy = this.getEventY(e, 1) - this.getEventY(e, 0);
      return Math.sqrt(dx * dx + dy * dy);
    }
  }

  loadImage(src: string) {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      this.initWebGL(image);
    };
    image.src = src;
  }

  initWebGL(image: HTMLImageElement) {
    const gl = this.gl;

    // Vertex shader
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

    // Fragment shader with a texture offset and scale
    const fragmentShaderSource = `#version 300 es

      precision highp float;
      
      in vec2 v_uv;
      out vec4 out_color;
      
      uniform sampler2D u_image;
      uniform int u_mode;
      uniform ivec2 u_self_tiles;
      uniform float u_wireframe;
      
      ${tiling_shader}
      
      void main() {
          out_color = textureTiling(u_image, v_uv, u_mode, u_self_tiles, u_wireframe, false);
      }
    `;
    const fragmentShader = this.createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);

    const program = this.createProgram(gl, vertexShader, fragmentShader);

    const positionLocation = gl.getAttribLocation(program, "a_position");
    const texCoordLocation = gl.getAttribLocation(program, "a_texCoord");
    for (const uniform of ['u_mins', 'u_maxs', 'u_mode', 'u_wireframe', 'u_self_tiles']) {
      this.uniforms.set(uniform, gl.getUniformLocation(program, uniform));
    }
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1.0, 1.0,
      1.0, 1.0,
      -1.0, -1.0,
      -1.0, -1.0,
      1.0, 1.0,
      1.0, -1.0,
    ]), gl.STATIC_DRAW);

    const texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
      0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
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
    gl.uniform2f(this.uniforms.get('u_mins'), this.minX, this.minY);
    gl.uniform2f(this.uniforms.get('u_maxs'), this.maxX, this.maxY);
    gl.uniform1i(this.uniforms.get('u_mode'), this.mode);
    gl.uniform2i(this.uniforms.get('u_self_tiles'), this.selfTilesX, this.selfTilesY);
    gl.uniform1f(this.uniforms.get('u_wireframe'), this.wireframe);
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

customElements.define('tiling-canvas', TilingCanvas);
