from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


def generate_comparison_html(img_before: Path, img_after: Path, out_html: Path, title: str = "MFAD Render Comparison") -> None:
    """Generate an interactive before/after split slider HTML viewer (Issue #8)."""
    # Use relative paths or data URIs for portability
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #121316;
      color: #e1e4ea;
      margin: 0;
      padding: 24px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    h1 {{
      margin-bottom: 8px;
      font-size: 24px;
      letter-spacing: -0.5px;
    }}
    .subtitle {{
      color: #8b949e;
      margin-bottom: 24px;
      font-size: 14px;
    }}
    .comparison-container {{
      position: relative;
      width: 800px;
      max-width: 95vw;
      aspect-ratio: 16 / 9;
      overflow: hidden;
      border-radius: 8px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.6);
      user-select: none;
      border: 1px solid #30363d;
    }}
    .comparison-img {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }}
    .img-overlay {{
      clip-path: inset(0 50% 0 0);
    }}
    .slider {{
      position: absolute;
      top: 0;
      bottom: 0;
      left: 50%;
      width: 4px;
      background: #58a6ff;
      cursor: ew-resize;
      z-index: 10;
      transform: translateX(-50%);
    }}
    .slider-handle {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 32px;
      height: 32px;
      background: #58a6ff;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 2px 6px rgba(0,0,0,0.5);
      color: #0d1117;
      font-weight: bold;
      font-size: 14px;
    }}
    .labels {{
      display: flex;
      justify-content: space-between;
      width: 800px;
      max-width: 95vw;
      margin-top: 12px;
      font-weight: 600;
      font-size: 14px;
    }}
    .badge {{
      padding: 4px 10px;
      border-radius: 4px;
      background: #21262d;
      border: 1px solid #30363d;
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="subtitle">Drag the slider horizontally to compare Direct Illumination vs Monte Carlo Path Tracing</div>

  <div class="comparison-container" id="container">
    <img src="{img_after.as_posix()}" class="comparison-img" alt="After (Path Tracing)" />
    <img src="{img_before.as_posix()}" class="comparison-img img-overlay" id="overlay" alt="Before (Direct Lighting)" />
    <div class="slider" id="slider">
      <div class="slider-handle">↔</div>
    </div>
  </div>

  <div class="labels">
    <span class="badge" style="color: #79c0ff;">← Direct Lighting (Whitted)</span>
    <span class="badge" style="color: #56d364;">Path Tracing (Global Illumination) →</span>
  </div>

  <script>
    const container = document.getElementById('container');
    const overlay = document.getElementById('overlay');
    const slider = document.getElementById('slider');
    let isDragging = false;

    function setSliderPos(x) {{
      const rect = container.getBoundingClientRect();
      let pos = (x - rect.left) / rect.width;
      pos = Math.max(0, Math.min(1, pos));
      const pct = (pos * 100).toFixed(2);
      overlay.style.clipPath = `inset(0 ${{100 - pct}}% 0 0)`;
      slider.style.left = `${{pct}}%`;
    }}

    container.addEventListener('mousedown', (e) => {{ isDragging = true; setSliderPos(e.clientX); }});
    window.addEventListener('mouseup', () => {{ isDragging = false; }});
    window.addEventListener('mousemove', (e) => {{ if (isDragging) setSliderPos(e.clientX); }});

    container.addEventListener('touchstart', (e) => {{ isDragging = true; setSliderPos(e.touches[0].clientX); }});
    window.addEventListener('touchend', () => {{ isDragging = false; }});
    window.addEventListener('touchmove', (e) => {{ if (isDragging) setSliderPos(e.touches[0].clientX); }});
  </script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def compute_difference_map(img1_path: Path, img2_path: Path, out_path: Path) -> float:
    """Compute per-pixel absolute difference and save difference heatmap (Issue #8)."""
    if Image is None:
        raise RuntimeError("Pillow is required for image comparison")

    im1 = np.array(Image.open(img1_path).convert("RGB"), dtype=np.float32) / 255.0
    im2 = np.array(Image.open(img2_path).convert("RGB"), dtype=np.float32) / 255.0

    if im1.shape != im2.shape:
        # Resize im2 to match im1
        pil2 = Image.open(img2_path).convert("RGB").resize((im1.shape[1], im1.shape[0]))
        im2 = np.array(pil2, dtype=np.float32) / 255.0

    diff = np.abs(im1 - im2)
    mae = float(np.mean(diff))

    # Boost difference for clear visualization (3x multiplier)
    vis = np.clip(diff * 3.0, 0.0, 1.0)
    diff_img = Image.fromarray((vis * 255.0).astype(np.uint8))
    diff_img.save(out_path)
    return mae


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Viewer: before/after image comparison tool (Issue #8)")
    parser.add_argument("img1", type=str, help="Path to first image (Before)")
    parser.add_argument("img2", type=str, help="Path to second image (After)")
    parser.add_argument("--diff", type=str, default="diff.png", help="Path to save difference map")
    parser.add_argument("--html", type=str, default="compare.html", help="Path to save interactive HTML slider")
    args = parser.parse_args(argv)

    p1, p2 = Path(args.img1), Path(args.img2)
    if not p1.exists() or not p2.exists():
        print(f"[ERROR] Image file not found: {p1} or {p2}", file=sys.stderr)
        return 1

    generate_comparison_html(p1, p2, Path(args.html))
    print(f"[SUCCESS] Interactive slider saved to {args.html}")

    try:
        mae = compute_difference_map(p1, p2, Path(args.diff))
        print(f"[SUCCESS] Difference heatmap saved to {args.diff} (Mean Absolute Difference: {mae:.5f})")
    except Exception as e:
        print(f"[WARN] Could not generate difference map: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
