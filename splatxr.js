let refSpace;
let gl;

const worldTransform = [
  0.9742308259010315, -0.05595879629254341, 0.21850138902664185, 0, -0.05095028132200241, -0.9982947111129761, -0.028494207188487053, 0,
  0.21972326934337616, 0.016627229750156403, -0.9754206538200378, 0, 0.21549134328961372, 1.9529861807823181, 1.4755789041519165, 1,
];

const vertexShaderSource = `
  #version 300 es
  precision highp float;
  precision highp int;
  
  uniform highp usampler2D u_texture;
  uniform mat4 projection, view;
  uniform vec2 focal;
  uniform vec2 viewport;
  uniform float time;
  
  in vec2 position;
  in int index;
  
  out vec4 vColor;
  out vec2 vPosition;
  
  void main () {
      gl_Position = vec4(0.0, 0.0, 2.0, 1.0);

      uvec4 motion1 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2) | 3u, uint(index) >> 10), 0);
      vec2 trbf = unpackHalf2x16(motion1.w);
      float dt = time - trbf.x;

      float topacity = exp(-1.0 * pow(dt / trbf.y, 2.0));
      if(topacity < 0.02) return;

      uvec4 motion0 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2) | 2u, uint(index) >> 10), 0);
      uvec4 static0 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2), uint(index) >> 10), 0);

      vec2 m0 = unpackHalf2x16(motion0.x), m1 = unpackHalf2x16(motion0.y), m2 = unpackHalf2x16(motion0.z), 
           m3 = unpackHalf2x16(motion0.w), m4 = unpackHalf2x16(motion1.x); 
      
      vec4 trot = vec4(unpackHalf2x16(motion1.y).xy, unpackHalf2x16(motion1.z).xy) * dt;
      vec3 tpos = (vec3(m0.xy, m1.x) * dt + vec3(m1.y, m2.xy) * dt*dt + vec3(m3.xy, m4.x) * dt*dt*dt);
      
      vec4 cam = view * vec4(uintBitsToFloat(static0.xyz) + tpos, 1);
      vec4 pos = projection * cam;
  
      float clip = 1.2 * pos.w;
      if (pos.z < -clip || pos.x < -clip || pos.x > clip || pos.y < -clip || pos.y > clip) return;
      uvec4 static1 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2) | 1u, uint(index) >> 10), 0);

      vec4 rot = vec4(unpackHalf2x16(static0.w).xy, unpackHalf2x16(static1.x).xy) + trot;
      vec3 scale = vec3(unpackHalf2x16(static1.y).xy, unpackHalf2x16(static1.z).x);
      rot /= sqrt(dot(rot, rot));
  
      mat3 S = mat3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z);
      mat3 R = mat3(
        1.0 - 2.0 * (rot.z * rot.z + rot.w * rot.w), 2.0 * (rot.y * rot.z - rot.x * rot.w), 2.0 * (rot.y * rot.w + rot.x * rot.z),
        2.0 * (rot.y * rot.z + rot.x * rot.w), 1.0 - 2.0 * (rot.y * rot.y + rot.w * rot.w), 2.0 * (rot.z * rot.w - rot.x * rot.y),
        2.0 * (rot.y * rot.w - rot.x * rot.z), 2.0 * (rot.z * rot.w + rot.x * rot.y), 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z));
      mat3 M = S * R;
      mat3 Vrk = 4.0 * transpose(M) * M;
      mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z), 
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z), 
        0., 0., 0.
    );
  
      mat3 T = transpose(mat3(view)) * J;
      mat3 cov2d = transpose(T) * Vrk * T;
  
      float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
      float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
      float lambda1 = mid + radius, lambda2 = mid - radius;
  
      if(lambda2 < 0.0) return;
      vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
      vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
      vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);
      
      uint rgba = static1.w;
      vColor = 
        clamp(pos.z/pos.w+1.0, 0.0, 1.0) * 
        vec4(1.0, 1.0, 1.0, topacity) *
        vec4(
          (rgba) & 0xffu, 
          (rgba >> 8) & 0xffu, 
          (rgba >> 16) & 0xffu, 
          (rgba >> 24) & 0xffu) / 255.0;

      vec2 vCenter = vec2(pos) / pos.w;
      gl_Position = vec4(
          vCenter 
          + position.x * majorAxis / viewport 
          + position.y * minorAxis / viewport, 0.0, 1.0);

      vPosition = position;
  }
  `.trim();

const fragmentShaderSource = `
  #version 300 es
  precision highp float;
  
  in vec4 vColor;
  in vec2 vPosition;
  
  out vec4 fragColor;
  
  void main () {
      float A = -dot(vPosition, vPosition);
      if (A < -4.0) discard;
      float B = exp(A) * vColor.a;
      fragColor = vec4(B * vColor.rgb, B);
  }
  
  `.trim();

const vertices = [
  [-1, -1, 0], // [x, y, z]
  [1, -1, 0],
  [1, 1, 0],
];

function createWorker(self) {
  let lastProj;
  let positions;
  let viewProj;
  let vertexCount;
  let lastVertexCount;

  self.onmessage = (e) => {
    if (e.data.view) {
      viewProj = e.data.view;
      throttledSort();
    } else if (e.data.texture) {
      texture = e.data.texture;
      vertexCount = Math.floor((texture.byteLength - e.data.remaining) / 4 / 16);
      if (vertexCount < 0) return;
      positions = new Float32Array(vertexCount * 3);
      for (let i = 0; i < vertexCount; i++) {
        positions[3 * i + 0] = texture[16 * i + 0];
        positions[3 * i + 1] = texture[16 * i + 1];
        positions[3 * i + 2] = texture[16 * i + 2];
      }
      throttledSort();
    }
  };

  let sortTimeout;
  function throttledSort() {
    if (!sortTimeout)
      sortTimeout = setTimeout(() => {
        runSort();
        sortTimeout = null;
      }, 0);
  }

  function runSort() {
    if (vertexCount < 0) return;
    if (!viewProj) return;
    if (lastProj && lastVertexCount === vertexCount) {
      let dist = Math.hypot(...[2, 6, 10].map((k) => lastProj[k] - viewProj[k]));
      if (dist < 0.01) return;
    }
    lastVertexCount = vertexCount;
    lastProj = viewProj;

    console.time("sort");
    let maxDepth = -Infinity;
    let minDepth = Infinity;
    let sizeList = new Int32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) {
      let depth =
        ((viewProj[2] * positions[3 * i + 0] + viewProj[6] * positions[3 * i + 1] + viewProj[10] * positions[3 * i + 2]) * 4096) | 0;
      sizeList[i] = depth;
      if (depth > maxDepth) maxDepth = depth;
      if (depth < minDepth) minDepth = depth;
    }
    // This is a 16 bit single-pass counting sort
    let depthInv = (256 * 256) / (maxDepth - minDepth);
    let counts0 = new Uint32Array(256 * 256);
    for (let i = 0; i < vertexCount; i++) {
      sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
      counts0[sizeList[i]]++;
    }
    let starts0 = new Uint32Array(256 * 256);
    for (let i = 1; i < 256 * 256; i++) starts0[i] = starts0[i - 1] + counts0[i - 1];
    depthIndex = new Uint32Array(vertexCount);
    for (let i = 0; i < vertexCount; i++) depthIndex[starts0[sizeList[i]]++] = i;
    console.timeEnd("sort");

    self.postMessage({ depthIndex, viewProj, vertexCount }, [depthIndex.buffer]);
  }
}

async function initXR() {
  let supported = navigator.xr && (await navigator.xr.isSessionSupported("immersive-vr"));
  console.log("Immersive VR supported", supported);

  let session = await navigator.xr.requestSession("immersive-vr");
  session.addEventListener("end", () => {
    console.log("end session");
  });
  const canvas = document.getElementById("canvas");
  gl = canvas.getContext("webgl2", {
    antialias: false,
    xrCompatible: true,
  });

  const program = attachShaders(gl, vertexShaderSource, fragmentShaderSource);

  gl.disable(gl.DEPTH_TEST); // Disable depth testing

  // Enable blending
  gl.enable(gl.BLEND);
  gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
  gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

  const u_projection = gl.getUniformLocation(program, "projection");
  const u_viewport = gl.getUniformLocation(program, "viewport");
  const u_focal = gl.getUniformLocation(program, "focal");
  const u_view = gl.getUniformLocation(program, "view");
  const u_time = gl.getUniformLocation(program, "time");

  // positions
  const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
  const vertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
  const a_position = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(a_position);
  gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
  gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

  var texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);

  var u_textureLocation = gl.getUniformLocation(program, "u_texture");
  gl.uniform1i(u_textureLocation, 0);

  const indexBuffer = gl.createBuffer();
  const a_index = gl.getAttribLocation(program, "index");
  gl.enableVertexAttribArray(a_index);
  gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
  gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
  gl.vertexAttribDivisor(a_index, 1);

  //   const vertexData = new Float32Array(vertices.flat());
  //   gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  //   gl.bufferData(gl.ARRAY_BUFFER, vertexData, gl.STATIC_DRAW);

  //   const vertexPosition = gl.getAttribLocation(program, "vertexPosition");
  //   gl.enableVertexAttribArray(vertexPosition);
  //   gl.vertexAttribPointer(vertexPosition, 3, gl.FLOAT, false, 0, 0);

  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });

  const worker = new Worker(
    URL.createObjectURL(
      new Blob(["(", createWorker.toString(), ")(self)"], {
        type: "application/javascript",
      })
    )
  );

  worker.onmessage = (e) => {
    if (e.data.depthIndex) {
      const { depthIndex, viewProj } = e.data;
      gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW);
      vertexCount = e.data.vertexCount;
      //   session.requestAnimationFrame(onXRFrame);
      console.log("depth sorted", vertexCount, depthIndex);
    }
  };

  refSpace = await session.requestReferenceSpace("local");

  const url = "model.splatv";
  const req = await fetch(url, { mode: "cors", credentials: "omit" });
  if (req.status != 200) throw new Error(req.status + " Unable to load " + req.url);

  let vertexCount = 0;
  let lastVertexCount = -1;
  const chunkHandler = (chunk, buffer, remaining, chunks) => {
    if (!remaining && chunk.type === "magic") {
      let intView = new Uint32Array(buffer);
      if (intView[0] !== 0x674b) throw new Error("This does not look like a splatv file");
      chunks.push({ size: intView[1], type: "chunks" });
    } else if (!remaining && chunk.type === "chunks") {
      for (let chunk of JSON.parse(new TextDecoder("utf-8").decode(buffer))) {
        chunks.push(chunk);
        if (chunk.type === "splat") {
          console.log(vertexCount);

          document.getElementById("spinner").style.display = "none";

          session.requestAnimationFrame(onXRFrame);
        }
      }
    } else if (chunk.type === "splat") {
      if (vertexCount > lastVertexCount || remaining === 0) {
        lastVertexCount = vertexCount;
        // vertexCount = Math.floor((buffer.byteLength - remaining) / 4 / 16);
        worker.postMessage({ texture: new Float32Array(buffer), remaining: remaining });
        const texdata = new Uint32Array(buffer);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, chunk.texwidth, chunk.texheight, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, texdata);
      }
    } else if (!remaining) {
      console.log("chunk", chunk, buffer);
    }
  };
  await readChunks(req.body.getReader(), [{ size: 8, type: "magic" }], chunkHandler);

  function onXRFrame(time, frame) {
    let session = frame.session;
    let pose = frame.getViewerPose(refSpace);
    if (!pose) return;
    let glLayer = session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, glLayer.framebuffer);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.uniform1f(u_time, Math.sin(Date.now() / 1000) / 2 + 1 / 2);

    for (let view of pose.views) {
      let viewport = glLayer.getViewport(view);
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
      let projectionMatrix = view.projectionMatrix;
      gl.uniform2fv(u_viewport, new Float32Array([viewport.width, viewport.height]));
      gl.uniformMatrix4fv(u_projection, false, projectionMatrix);

      gl.uniform2fv(u_focal, new Float32Array([(projectionMatrix[0] * viewport.width) / 2, -(projectionMatrix[5] * viewport.height) / 2]));
      const transformedView = multiply4(view.transform.inverse.matrix, worldTransform);
      gl.uniformMatrix4fv(u_view, false, transformedView);

      gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);

      const viewProj = multiply4(projectionMatrix, transformedView);
      worker.postMessage({ view: viewProj });
    }
    session.requestAnimationFrame(onXRFrame);
  }
}

function attachShaders(gl, vertexShaderSource, fragmentShaderSource) {
  const vertexShader = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vertexShader, vertexShaderSource);
  gl.compileShader(vertexShader);
  if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(vertexShader));

  const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fragmentShader, fragmentShaderSource);
  gl.compileShader(fragmentShader);
  if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(fragmentShader));

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.useProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) console.error(gl.getProgramInfoLog(program));
  return program;
}

async function readChunks(reader, chunks, handleChunk) {
  let chunk = chunks.shift();
  let buffer = new Uint8Array(chunk.size);
  let offset = 0;
  while (chunk) {
    let { done, value } = await reader.read();
    if (done) break;
    while (value.length + offset >= chunk.size) {
      buffer.set(value.subarray(0, chunk.size - offset), offset);
      value = value.subarray(chunk.size - offset);
      handleChunk(chunk, buffer.buffer, 0, chunks);
      chunk = chunks.shift();
      if (!chunk) break;
      buffer = new Uint8Array(chunk.size);
      offset = 0;
    }
    if (!chunk) break;
    buffer.set(value, offset);
    offset += value.length;
    handleChunk(chunk, buffer.buffer, buffer.byteLength - offset, chunks);
  }
  if (chunk) handleChunk(chunk, buffer.buffer, 0, chunks);
}

document.getElementById("enter").onclick = () => {
  initXR().catch((err) => {
    alert(err);
  });
};

initXR().catch((err) => {
  document.getElementById("spinner").style.cursor = "pointer";
  document.getElementById("spinner").onclick = () => {
    initXR().catch((err) => {
      alert(err);
    });
  };
});

function multiply4(a, b) {
  return [
    b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
    b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
    b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
    b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
    b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
    b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
    b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
    b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
    b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
    b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
    b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
    b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
    b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
    b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
    b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
    b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
  ];
}
