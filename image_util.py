import numpy as np
from PIL import Image


def average_blend(thumbnails: list[Image.Image]) -> Image.Image:
    stack = np.stack([np.array(img, dtype=np.float32) for img in thumbnails], axis=0)
    return Image.fromarray(np.mean(stack, axis=0).astype(np.uint8))


def median_blend(thumbnails: list[Image.Image]) -> Image.Image:
    stack = np.stack([np.array(img, dtype=np.float32) for img in thumbnails], axis=0)
    return Image.fromarray(np.median(stack, axis=0).astype(np.uint8))


def geometric_mean_blend(thumbnails: list[Image.Image]) -> Image.Image:
    stack = np.stack([np.array(img, dtype=np.float32) + 1 for img in thumbnails], axis=0)
    geo_mean = np.exp(np.mean(np.log(stack), axis=0)) - 1
    return Image.fromarray(np.clip(geo_mean, 0, 255).astype(np.uint8))


def overlay_blend(thumbnails: list[Image.Image]) -> Image.Image:
    stack = np.stack([np.array(img, dtype=np.float32) / 255 for img in thumbnails], axis=0)
    avg, multiplied = np.mean(stack, axis=0), np.prod(stack, axis=0)
    mask = avg <= 0.5
    overlay = np.zeros_like(avg)
    overlay[mask] = 2 * multiplied[mask] * avg[mask]
    overlay[~mask] = 1 - 2 * (1 - multiplied[~mask]) * (1 - avg[~mask])
    return Image.fromarray((overlay * 255).clip(0, 255).astype(np.uint8))


def max_blend(thumbnails: list[Image.Image]) -> Image.Image:
    stack = np.stack([np.array(img, dtype=np.uint8) for img in thumbnails], axis=0)
    return Image.fromarray(np.max(stack, axis=0))


def min_blend(thumbnails: list[Image.Image]) -> Image.Image:
    stack = np.stack([np.array(img, dtype=np.uint8) for img in thumbnails], axis=0)
    return Image.fromarray(np.min(stack, axis=0))


black_threshold = 30
consecutive_rows = 3


def crop_black_bars(img: Image.Image):
    arr = np.array(img.convert('L'))
    h, w = arr.shape

    top, bottom = 0, h
    for i in range(h):
        if all(is_black_row(arr[j]) for j in range(i, min(i + consecutive_rows, h))):
            top = i + consecutive_rows
        else:
            break
    for i in range(h - 1, -1, -1):
        if all(is_black_row(arr[j]) for j in range(max(0, i - consecutive_rows + 1), i + 1)):
            bottom = i - consecutive_rows + 1
        else:
            break

    left, right = 0, w
    for i in range(w):
        if all(is_black_row(arr[:, j]) for j in range(i, min(i + consecutive_rows, w))):
            left = i + consecutive_rows
        else:
            break
    for i in range(w - 1, -1, -1):
        if all(is_black_row(arr[:, j]) for j in range(max(0, i - consecutive_rows + 1), i + 1)):
            right = i - consecutive_rows + 1
        else:
            break

    if top >= bottom or left >= right:
        return img
    return img.crop((left, top, right, bottom))


def is_black_row(row: np.ndarray) -> bool:
    return np.mean(row) < black_threshold


def resize_images(images: list[Image.Image], thumbnail_resolution: tuple[int, int] | list[int]) -> list[Image.Image]:
    resized_images: list[Image.Image] = []

    for image in images:
        try:
            image = image.resize(thumbnail_resolution)
            resized_images.append(image)
        except Exception as e:
            print(f"(╥‸╥) error resizing image: {e}")

    return resized_images
