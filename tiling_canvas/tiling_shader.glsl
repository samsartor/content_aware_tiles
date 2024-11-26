vec4 blend(vec4 b, vec4 a) {
  return vec4(a.rgb * a.a + b.rgb * b.a * (1.0f - a.a), a.a + b.a * (1.0f - a.a));
}

uint lowbias32(uint x) {
  x ^= x >> 17;
  x *= 0xed5ad4bbU;
  x ^= x >> 11;
  x *= 0xac4c1b51U;
  x ^= x >> 15;
  x *= 0x31848babU;
  x ^= x >> 14;
  return x;
}

int tileEdgeRandom(int x, int y, uint div, uint rem) {
  uint i = lowbias32(lowbias32(uint(x)) + uint(y));
  return int((i / div) % rem);
}

struct TileColors {
  int l;
  int t;
  int r;
  int b;
};

TileColors interiorTileColors(vec2 i, uint colors) {
  int l = tileEdgeRandom(int(i.x), int(i.y), 1U, colors);
  int t = tileEdgeRandom(int(i.x), int(i.y), colors, colors);
  int r = tileEdgeRandom(int(i.x) + 1, int(i.y), 1U, colors);
  int b = tileEdgeRandom(int(i.x), int(i.y) + 1, colors, colors);
  return TileColors(l, t, r, b);
}

TileColors crossTileColors(vec2 i, uint colors) {
  int l = tileEdgeRandom(int(i.x) - 1, int(i.y), colors, colors);
  int t = tileEdgeRandom(int(i.x), int(i.y) - 1, 1U, colors);
  int r = tileEdgeRandom(int(i.x), int(i.y), colors, colors);
  int b = tileEdgeRandom(int(i.x), int(i.y), 1U, colors);
  return TileColors(l, t, r, b);
}

struct Details {
  ivec2 selfTiles;
  int selfOffset;
  float wireframeAlpha;
  float wireframeThickness;
  bool flipY;
};

vec4 tileWireframe(TileColors c, vec2 f, int mode, Details details) {
  vec3 colors[6] = vec3[](vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 1.0f, 1.0f), vec3(1.0f, 0.0f, 1.0f), vec3(1.0f, 1.0f, 0.0f));
  float thick = details.wireframeThickness;
  float thin = 1.0 - thick;
  if (mode == 0 || mode == 1) {
    float r = max(abs(f.x), abs(f.y));
    if(r > thin && r < 1.0 + thick) {
      return vec4(1.0f, 1.0f, 1.0f, details.wireframeAlpha);
    }
  } else if (mode == 2) {
    float r = max(abs(f.x), abs(f.y));
    if(f.x <= f.y && f.x < -f.y && r > thin && r < 1.0 + thick) {
        return vec4(colors[c.l], details.wireframeAlpha);
    }
    if(f.x >= f.y && f.x > -f.y && r > thin && r < 1.0 + thick) {
        return vec4(colors[c.r], details.wireframeAlpha);
    }
    if(f.x > f.y && f.x <= -f.y && r > thin && r < 1.0 + thick) {
        return vec4(colors[c.t], details.wireframeAlpha);
    }
    if(f.x < f.y && f.x >= -f.y && r > thin && r < 1.0 + thick) {
        return vec4(colors[c.b], details.wireframeAlpha);
    }
  } else if (mode == 3) {
    float tdist = length(f - vec2( 0.0f, -1.0f));
    float bdist = length(f - vec2( 0.0f,  1.0f));
    float ldist = length(f - vec2(-1.0f, 0.0f));
    float rdist = length(f - vec2( 1.0f,  0.0f));

    if (tdist < thick && tdist < bdist && tdist < ldist && tdist < rdist) {
      return vec4(colors[c.t], details.wireframeAlpha);
    }
    if (ldist < thick && ldist < bdist && ldist < tdist && ldist < rdist) {
      return vec4(colors[c.l], details.wireframeAlpha);
    }
    if (bdist < thick && bdist < ldist && bdist < tdist && bdist < rdist) {
      return vec4(colors[c.b], details.wireframeAlpha);
    }
    if (rdist < thick && rdist < ldist && rdist < tdist && rdist < bdist) {
      return vec4(colors[c.r], details.wireframeAlpha);
    }
  }
  return vec4(0.0f, 0.0f, 0.0f, 0.0f);
}

const int INVERSE_PACKING[162] = int[](0, 60, 40, 57, 27, 16, 41, 11, 72, 50, 10, 70, 26, 67, 37, 1, 51, 21, 20, 30, 80, 77, 6, 47, 61, 71, 31, 49, 19, 8, 17, 68, 48, 65, 44, 24, 9, 59, 29, 58, 18, 78, 34, 75, 45, 69, 79, 39, 28, 38, 7, 4, 14, 55, 25, 76, 56, 33, 3, 64, 73, 52, 32, 66, 35, 5, 74, 43, 13, 42, 2, 62, 36, 46, 15, 53, 54, 23, 12, 22, 63, 131, 90, 147, 101, 150, 117, 81, 130, 106, 107, 139, 155, 158, 109, 134, 138, 98, 114, 82, 115, 123, 142, 85, 93, 122, 146, 154, 91, 140, 116, 141, 100, 157, 111, 160, 127, 148, 99, 124, 108, 149, 84, 87, 119, 135, 132, 156, 83, 92, 125, 133, 152, 95, 103, 151, 110, 86, 161, 120, 96, 121, 89, 137, 118, 159, 94, 128, 88, 104, 97, 129, 145, 102, 126, 143, 112, 136, 144, 153, 105, 113);

int tileIndex(TileColors neighbors, int offset) {
  int tile = neighbors.l + neighbors.t * 3 + neighbors.r * 9 + neighbors.b * 27 + offset;
  return int(INVERSE_PACKING[tile]);
}

vec4 textureTiling(sampler2D image, vec2 uv, int mode, Details details) {
  vec2 f = fract(uv);
  vec2 i = uv - f;
  f = f * 2.0f - 1.0f;

  TileColors neighbors;
  vec2 tile_uv;
  vec2 par_uv = uv;

  vec2 selfTilesF = vec2(float(details.selfTiles.x), float(details.selfTiles.y));

  if (mode == 0 || mode == 1) {
    // Stochastic Self-tiling Tiles
    neighbors = TileColors(0, 0, 0, 0);
    int index;
    if (mode == 1) {
      index = tileEdgeRandom(int(i.x), int(i.y), 1U, uint(details.selfTiles.x * details.selfTiles.y));
    } else {
      index = 0;
    }
    index += details.selfOffset;
    tile_uv = f * 0.5f + 0.5f;
    tile_uv += vec2(float(index % details.selfTiles.x), float(index / details.selfTiles.x));
    tile_uv /= selfTilesF;
    par_uv /= selfTilesF;
  } else {
    // Wang Tiles (or Interior Tiles)
    neighbors = interiorTileColors(i, 3U);
    int index = tileIndex(neighbors, 0);

    tile_uv = f * 0.5f + 0.5f;

    vec2 cross_uv = uv + 0.5;
    vec2 cross_f = fract(cross_uv);
    vec2 cross_i = cross_uv - cross_f;
    cross_f = cross_f * 2.0 - 1.0;
    if(mode == 3 && abs(cross_f.x) + abs(cross_f.y) < abs(f.x) + abs(f.y)) {
      // Cross Tiles
      f = cross_f;
      tile_uv = f * 0.5f;
      neighbors = crossTileColors(cross_i, 3U);
      index = tileIndex(neighbors, 81);
    }

    tile_uv += vec2(float(index % 9), float(index / 9));
    tile_uv /= 9.0f;
    par_uv /= 9.0f;
  }

  if (details.flipY) {
    tile_uv.y = 1.0 - tile_uv.y;
  }

  vec4 wireframe_color = tileWireframe(neighbors, f, mode, details);
  vec4 texture_color = textureGrad(image, tile_uv, dFdx(par_uv), dFdy(par_uv));
  return blend(texture_color, wireframe_color);
}
