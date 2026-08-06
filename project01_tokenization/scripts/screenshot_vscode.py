"""Capture the real VS Code window, and nothing that happens to be sitting on top of it.

Two traps this avoids, both of which have bitten before:

1. DPI scaling -- without SetProcessDPIAware() the window rectangle comes back in
   logical pixels while the bitmap is in physical pixels, and you get a magnified crop
   of the top-left corner.
2. Overlapping windows -- CopyFromScreen grabs whatever is on screen, which on a
   previous run meant publishing a chat window and a browser to a public repo.
   PrintWindow(hwnd, hdc, PW_RENDERFULLCONTENT) asks the window to render ITSELF, so
   anything covering it is irrelevant.

Usage:
    python -m scripts.screenshot_vscode <output.png> [title-substring] [crop-width-px]

`crop-width-px` keeps only the leftmost N pixels. Use it to drop the secondary side
bar (the extension/chat panel) before publishing -- it is never part of the result
being shown, and panels are exactly the thing you do not want to push to a public repo.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

from PIL import Image

PW_RENDERFULLCONTENT = 2

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32


class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


def find_window(needle: str) -> tuple[int, str]:
    """Newest visible top-level window whose title contains `needle`."""
    found: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n == 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if needle.lower() in buf.value.lower():
            found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(cb, 0)
    if not found:
        raise SystemExit(f"no visible window matching {needle!r}. Is VS Code open?")
    return found[0]


def capture(hwnd: int) -> Image.Image:
    rect = RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w <= 0 or h <= 0:
        raise SystemExit(f"window has no area: {w}x{h}")

    src = user32.GetWindowDC(hwnd)
    dc = gdi32.CreateCompatibleDC(src)
    bmp = gdi32.CreateCompatibleBitmap(src, w, h)
    gdi32.SelectObject(dc, bmp)

    ok = user32.PrintWindow(hwnd, dc, PW_RENDERFULLCONTENT)

    buf = ctypes.create_string_buffer(w * h * 4)
    header = BITMAPINFOHEADER()
    header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    header.biWidth = w
    header.biHeight = -h        # negative => top-down rows, matching PIL's order
    header.biPlanes = 1
    header.biBitCount = 32
    header.biCompression = 0    # BI_RGB
    gdi32.GetDIBits(dc, bmp, 0, h, buf, ctypes.byref(header), 0)

    img = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB")

    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(dc)
    user32.ReleaseDC(hwnd, src)

    if not ok:
        print("  warning: PrintWindow returned 0; the image may be incomplete")
    return img


def main() -> None:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware, before any
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/vscode.png")
    needle = sys.argv[2] if len(sys.argv) > 2 else "Visual Studio Code"

    hwnd, title = find_window(needle)
    print(f"  window: {title!r}  (hwnd={hwnd})")
    img = capture(hwnd)

    if len(sys.argv) > 3:
        keep = int(sys.argv[3])
        print(f"  cropping {img.width} -> {keep} px wide (dropping the side panel)")
        img = img.crop((0, 0, min(keep, img.width), img.height))

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"  wrote {out}  ({img.width}x{img.height})")
    print(f"  INSPECT THIS IMAGE before committing it.")


if __name__ == "__main__":
    main()
