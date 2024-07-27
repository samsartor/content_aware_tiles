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

vec4 tileWireframe(TileColors c, vec2 f, float alpha) {
  vec3 colors[6] = vec3[](vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 1.0f, 1.0f), vec3(1.0f, 0.0f, 1.0f), vec3(1.0f, 1.0f, 0.0f));
  if(f.x <= f.y && f.x < 1.0f - f.y && f.x < 0.05f) {
      return vec4(colors[c.l], alpha);
  }
  if(f.x >= f.y && f.x > 1.0f - f.y && f.x > 0.95f) {
      return vec4(colors[c.r], alpha);
  }
  if(f.x > f.y && f.x <= 1.0f - f.y && f.y < 0.05f) {
      return vec4(colors[c.t], alpha);
  }
  if(f.x < f.y && f.x >= 1.0f - f.y && f.y > 0.95f) {
      return vec4(colors[c.b], alpha);
  }
  return vec4(0.0f, 0.0f, 0.0f, 0.0f);
}

int INVERSE_PACKING[162] = int[](0, 60, 40, 57, 27, 16, 41, 11, 72, 50, 10, 70, 26, 67, 37, 1, 51, 21, 20, 30, 80, 77, 6, 47, 61, 71, 31, 49, 19, 8, 17, 68, 48, 65, 44, 24, 9, 59, 29, 58, 18, 78, 34, 75, 45, 69, 79, 39, 28, 38, 7, 4, 14, 55, 25, 76, 56, 33, 3, 64, 73, 52, 32, 66, 35, 5, 74, 43, 13, 42, 2, 62, 36, 46, 15, 53, 54, 23, 12, 22, 63, 131, 90, 147, 101, 150, 117, 81, 130, 106, 107, 139, 155, 158, 109, 134, 138, 98, 114, 82, 115, 123, 142, 85, 93, 122, 146, 154, 91, 140, 116, 141, 100, 157, 111, 160, 127, 148, 99, 124, 108, 149, 84, 87, 119, 135, 132, 156, 83, 92, 125, 133, 152, 95, 103, 151, 110, 86, 161, 120, 96, 121, 89, 137, 118, 159, 94, 128, 88, 104, 97, 129, 145, 102, 126, 143, 112, 136, 144, 153, 105, 113);

int tileIndex(TileColors neighbors, int offset) {
  int tile = neighbors.l + neighbors.t * 3 + neighbors.r * 9 + neighbors.b * 27 + offset;
  return int(INVERSE_PACKING[tile]);
}

vec4 textureTiling(sampler2D image, vec2 uv, int mode, ivec2 self_tiles, float wireframe, bool flip_y) {
  vec2 f = fract(uv);
  vec2 i = uv - f;

  TileColors neighbors;
  vec2 tile_uv;
  vec2 par_uv = uv;

  if(mode == 0) {
      // Single Tile
      neighbors = TileColors(0, 0, 0, 0);
      tile_uv = f;
  } else if(mode == 1) {
      // Stochastic Self-tiling Tiles
      neighbors = TileColors(0, 0, 0, 0);
      int index = tileEdgeRandom(int(i.x), int(i.y), 1U, uint(self_tiles.x * self_tiles.y));
      tile_uv = vec2(f.x + float(index % self_tiles.x), f.y + float(index / self_tiles.x)) / float(self_tiles);
      par_uv /= float(self_tiles);
  } else {
      // Wang Tiles (or Interior Tiles)
      neighbors = interiorTileColors(i, 3U);
      int index = tileIndex(neighbors, 0);
      tile_uv = vec2(f.x + float(index % 9), f.y + float(index / 9)) / 9.0f;
      par_uv /= 9.0f;

      vec2 cross_uv = uv + 0.5;
      vec2 cross_f = fract(cross_uv);
      if(mode == 3 && abs(cross_f.x - 0.5) + abs(cross_f.y - 0.5) < 0.5) {
          // Cross Tiles
          vec2 cross_i = cross_uv - cross_f;
          TileColors cross_neighbors = crossTileColors(cross_i, 3U);
          int cross_index = tileIndex(cross_neighbors, 81);
          tile_uv = vec2(cross_f.x + float(cross_index % 9) - 0.5, cross_f.y + float(cross_index / 9) - 0.5) / 9.0f;
      }
  }

  if (flip_y) {
      tile_uv.y = 1.0 - tile_uv.y;
  }

  vec4 wireframe_color = tileWireframe(neighbors, f, wireframe);
  vec4 texture_color = textureGrad(image, tile_uv, dFdx(par_uv), dFdy(par_uv));
  return mix(texture_color, wireframe_color, wireframe_color.w);
}
