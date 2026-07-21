from pathlib import Path

import cv2


def decode(path: Path, output_dir: Path, step: int = 1) -> int:
    """Decode a video into frames and save them as images."""
    path = Path(path)
    output_dir = Path(output_dir)

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    frame_index = 0
    saved = 0
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            if frame_index % step == 0:
                frame_path = output_dir / f"frame_{frame_index:06d}.png"
                cv2.imwrite(str(frame_path), frame)
                saved += 1

            frame_index += 1
    finally:
        capture.release()

    return saved
