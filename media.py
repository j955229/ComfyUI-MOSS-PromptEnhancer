from __future__ import annotations

import base64
import io
import re


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


def image_input_index(name):
    """The original image socket is image1; extra sockets are image2, image3, ..."""
    if name == "image":
        return 1
    match = re.fullmatch(r"image([1-9][0-9]*)", name) if isinstance(name, str) else None
    return int(match.group(1)) if match else None


def collect_images(image, extra_inputs):
    """Keep socket order and the existing first-frame-per-socket behavior."""
    sources = {"image": image, **extra_inputs}
    unknown = [name for name in extra_inputs if image_input_index(name) is None]
    if unknown:
        raise TypeError(f"Unknown image inputs: {', '.join(unknown)}")
    images = []
    seen = set()
    for name in sorted(sources, key=image_input_index):
        value = sources[name]
        if value is None:
            continue
        index = image_input_index(name)
        if index in seen:
            raise ValueError(f"image{index} 重复；第一张图片请只使用 image 入口。")
        seen.add(index)
        width, height, batch = image_metadata(value)
        images.append({
            "name": f"image{index}", "tensor": value[:1, ..., :3],
            "width": width, "height": height, "batch": batch, "frame_index": 0,
        })
    return images


def clip_image_batch(images, max_edge=1536):
    """Batch separate references without stretching or cropping their contents."""
    if not images:
        return None
    if len(images) == 1:
        return images[0]
    import torch
    import torch.nn.functional as F

    height = max(int(image.shape[1]) for image in images)
    width = max(int(image.shape[2]) for image in images)
    scale = min(1.0, max_edge / max(height, width))
    height, width = max(1, round(height * scale)), max(1, round(width * scale))
    first = images[0]
    frames = []
    for image in images:
        image = image.to(device=first.device, dtype=first.dtype).movedim(-1, 1)
        h, w = image.shape[-2:]
        fit = min(height / h, width / w)
        out_h, out_w = max(1, round(h * fit)), max(1, round(w * fit))
        if (h, w) != (out_h, out_w):
            image = F.interpolate(image, size=(out_h, out_w), mode="bilinear", align_corners=False)
        top, left = (height - out_h) // 2, (width - out_w) // 2
        image = F.pad(image, (left, width - out_w - left, top, height - out_h - top), value=0.5)
        frames.append(image.movedim(1, -1))
    return torch.cat(frames, dim=0)
