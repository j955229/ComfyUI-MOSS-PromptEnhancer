from __future__ import annotations

import base64
import io


def image_metadata(image) -> tuple[int, int, int]:
    """Return (width, height, batch) for a ComfyUI IMAGE tensor."""
    shape = tuple(int(value) for value in image.shape)
    if len(shape) != 4:
        raise ValueError(f"IMAGE must have shape [batch, height, width, channels]; got {shape}.")
    batch, height, width, channels = shape
    if batch < 1 or height < 1 or width < 1:
        raise ValueError(f"IMAGE must not be empty; got shape {shape}.")
    if channels < 3:
        raise ValueError(f"IMAGE must have at least 3 color channels; got {channels}.")
    return width, height, batch


def prepare_image(image, max_edge: int = 1536, frame_index: int = 0) -> dict[str, object]:
    """Convert one ComfyUI IMAGE frame to an inline image payload for llama-server."""
    import numpy as np
    from PIL import Image

    width, height, batch = image_metadata(image)
    if frame_index < 0 or frame_index >= batch:
        raise ValueError(f"frame_index {frame_index} is outside the IMAGE batch of {batch}.")

    frame = image[frame_index, ..., :3]
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().float().numpy()
    else:
        frame = np.asarray(frame, dtype=np.float32)
    frame = np.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=0.0)
    pixels = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(pixels, mode="RGB")
    original_size = pil_image.size
    if max_edge > 0:
        pil_image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=95, subsampling=0, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "data_url": f"data:image/jpeg;base64,{encoded}",
        "width": width,
        "height": height,
        "batch": batch,
        "frame_index": frame_index,
        "original_size": original_size,
        "encoded_size": pil_image.size,
    }
