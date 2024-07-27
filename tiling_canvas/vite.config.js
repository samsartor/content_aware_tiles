import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    minify: false,
    lib: {
      entry: './tiling_canvas.ts',
      formats: ['es'],
    },
  }
});
