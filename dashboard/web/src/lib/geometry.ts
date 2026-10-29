export type Pt = [number, number];

/**
 * Homography (3x3, row-major, h33 = 1) mapping the 4 `src` points onto the 4 `dst` points,
 * or null if the points are degenerate (collinear / duplicated).
 */
export function solveHomography(src: Pt[], dst: Pt[]): number[] | null {
  if (src.length !== 4 || dst.length !== 4) return null;
  const A: number[][] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y, u]);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y, v]);
  }
  // Gaussian elimination with partial pivoting on the 8x9 augmented matrix.
  for (let col = 0; col < 8; col++) {
    let pivot = col;
    for (let r = col + 1; r < 8; r++) if (Math.abs(A[r][col]) > Math.abs(A[pivot][col])) pivot = r;
    if (Math.abs(A[pivot][col]) < 1e-9) return null;
    [A[col], A[pivot]] = [A[pivot], A[col]];
    for (let r = 0; r < 8; r++) {
      if (r === col) continue;
      const f = A[r][col] / A[col][col];
      for (let c = col; c < 9; c++) A[r][c] -= f * A[col][c];
    }
  }
  const h = A.map((row, i) => row[8] / row[i]);
  return h.every(Number.isFinite) ? [...h, 1] : null;
}

export function applyHomography(h: number[], [x, y]: Pt): Pt {
  const w = h[6] * x + h[7] * y + h[8];
  return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w];
}

/** Arrow from the middle of a stop line toward the side vehicles must travel to count (matches the Python tools). */
export function directionArrow(a: Pt, b: Pt, direction: "positive" | "negative", length: number): [Pt, Pt] {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const norm = Math.hypot(dx, dy) || 1;
  const sign = direction === "positive" ? 1 : -1;
  const mid: Pt = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  return [mid, [mid[0] + (sign * -dy / norm) * length, mid[1] + (sign * dx / norm) * length]];
}
