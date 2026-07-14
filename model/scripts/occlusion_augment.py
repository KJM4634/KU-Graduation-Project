"""
합성 Occlusion Augmentation 파이프라인 (PRD 7.2절)

절차적으로 생성한 잔해 텍스처(콘크리트/철근/먼지/천막)를 사람 bbox 영역 하단부에 합성하여
occlusion 비율(30/50/70%)을 제어한다. 원본 라벨은 full-extent bbox를 그대로 유지한다
(가려짐과 무관하게 사람의 실제 전체 범위를 라벨링 — PRD 6장/8장).

입력: YOLO 포맷 데이터셋 (images/, labels/ — 각 라벨은 "class xc yc w h" 정규화 좌표)
출력: model/data/synthetic/images, labels, metadata.csv

사용 예:
    python occlusion_augment.py \
        --images model/data/processed/coco/images \
        --labels model/data/processed/coco/labels \
        --out model/data/synthetic \
        --ratios 0.3 0.5 0.7 \
        --variants-per-image 1
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TEXTURE_TYPES = ["concrete", "rebar", "dust", "tarp"]
TOLERANCE = 0.05  # PRD 7.2: 실측 비율 ±5% 허용오차
MAX_ATTEMPTS = 12  # 목표 비율에 수렴시키기 위한 최대 재시도 횟수


def read_yolo_labels(label_path, img_w, img_h):
    """YOLO 정규화 라벨을 픽셀 좌표 [x1,y1,x2,y2] 리스트로 변환."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls, xc, yc, w, h = (float(v) for v in line.split())
        xc, yc, w, h = xc * img_w, yc * img_h, w * img_w, h * img_h
        x1, y1 = xc - w / 2, yc - h / 2
        x2, y2 = xc + w / 2, yc + h / 2
        boxes.append((int(cls), max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)))
    return boxes


def _perlin_like_noise(w, h, scale=8, seed=None):
    """의사 Perlin 노이즈: 저해상도 랜덤값을 bicubic 업샘플링해 자연스러운 얼룩 패턴 생성."""
    rng = np.random.default_rng(seed)
    small = rng.random((max(2, h // scale), max(2, w // scale)))
    noise_img = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(noise_img, dtype=np.float32) / 255.0


def generate_debris_texture(width, height, texture_type, seed=None):
    """
    잔해 텍스처(RGBA)를 절차적으로 생성. 위쪽 경계는 들쭉날쭉하게(잔해 더미처럼) 처리.
    반환: (RGBA 이미지, 상단 경계의 열별 높이 배열(0~height, 값이 클수록 더 많이 덮임))
    """
    width, height = max(1, width), max(1, height)
    rng = np.random.default_rng(seed)
    noise = _perlin_like_noise(width, height, scale=max(4, width // 12), seed=seed)

    if texture_type == "concrete":
        base = np.array([150, 148, 142], dtype=np.float32)
        variation = (noise[..., None] - 0.5) * 60
    elif texture_type == "rebar":
        base = np.array([120, 110, 95], dtype=np.float32)
        variation = (noise[..., None] - 0.5) * 50
    elif texture_type == "dust":
        base = np.array([180, 170, 150], dtype=np.float32)
        variation = (noise[..., None] - 0.5) * 40
    else:  # tarp
        base = np.array([70, 90, 100], dtype=np.float32)
        variation = (noise[..., None] - 0.5) * 30

    rgb = np.clip(base + variation, 0, 255).astype(np.uint8)
    rgb_img = Image.fromarray(np.repeat(rgb, 1, axis=2) if rgb.shape[-1] == 3 else rgb, mode="RGB")

    if texture_type == "rebar":
        draw = ImageDraw.Draw(rgb_img)
        spacing = max(6, width // 6)
        for x in range(0, width, spacing):
            draw.line([(x, 0), (x, height)], fill=(90, 85, 80), width=max(1, width // 60))
        for y in range(0, height, spacing):
            draw.line([(0, y), (width, y)], fill=(90, 85, 80), width=max(1, height // 60))

    # 상단 경계를 노이즈 기반으로 들쭉날쭉하게 (잔해 더미가 사람을 불규칙하게 덮는 느낌)
    edge_noise = _perlin_like_noise(width, 1, scale=max(2, width // 10), seed=None if seed is None else seed + 1)[0]
    jagged_top = (edge_noise * height * 0.35).astype(np.int32)

    alpha = np.zeros((height, width), dtype=np.uint8)
    for x in range(width):
        top = min(height, jagged_top[x])
        alpha[top:, x] = 235 if texture_type != "tarp" else 210

    rgba = np.dstack([np.asarray(rgb_img), alpha])
    texture_img = Image.fromarray(rgba, mode="RGBA")
    texture_img = texture_img.filter(ImageFilter.GaussianBlur(radius=1))
    return texture_img, jagged_top


def apply_occlusion(image, box, target_ratio, rng, tolerance=TOLERANCE, max_attempts=MAX_ATTEMPTS):
    """
    bbox 하단부에서부터 잔해 텍스처를 합성해 target_ratio(occlusion 비율)에 수렴시킨다.
    반환: (occluded PIL.Image, 실제 occlusion 비율, 사용한 texture_type)
    """
    _, x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 1 or box_h <= 1:
        return image, 0.0, "none"

    texture_type = rng.choice(TEXTURE_TYPES)
    result = image.copy()

    # 덮는 높이 비율(occlusion_height_ratio)을 이분 탐색으로 조정해 실제 픽셀 기준
    # occlusion 비율이 target_ratio에 ±tolerance 이내로 수렴하도록 함.
    lo, hi = 0.0, 1.0
    best_img, best_ratio = image, 0.0
    for attempt in range(max_attempts):
        cover_ratio = (lo + hi) / 2
        cover_h = max(1, int(box_h * cover_ratio))
        seed = rng.integers(0, 2**31 - 1)
        texture, _ = generate_debris_texture(box_w, cover_h, texture_type, seed=seed)

        crop = result.crop((x1, y2 - cover_h, x2, y2)).convert("RGBA")
        composited = Image.alpha_composite(crop, texture).convert("RGB")
        candidate = result.copy()
        candidate.paste(composited, (x1, y2 - cover_h))

        alpha_arr = np.asarray(texture)[..., 3]
        actual_ratio = float((alpha_arr > 0).sum()) / (box_w * box_h)

        if abs(actual_ratio - target_ratio) < abs(best_ratio - target_ratio) or attempt == 0:
            best_img, best_ratio = candidate, actual_ratio

        if abs(actual_ratio - target_ratio) <= tolerance:
            return candidate, actual_ratio, texture_type

        if actual_ratio < target_ratio:
            lo = cover_ratio
        else:
            hi = cover_ratio

    return best_img, best_ratio, texture_type


def process_dataset(images_dir, labels_dir, out_dir, ratios, variants_per_image, seed=0):
    images_dir, labels_dir, out_dir = Path(images_dir), Path(labels_dir), Path(out_dir)
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / "metadata.csv"
    write_header = not metadata_path.exists()
    rng = np.random.default_rng(seed)

    image_paths = sorted(
        p for p in images_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not image_paths:
        print(f"경고: {images_dir} 에서 이미지를 찾지 못함")
        return

    with open(metadata_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "source_image", "output_image", "bbox_index",
                "target_ratio", "actual_ratio", "texture_type",
            ])

        for img_path in image_paths:
            label_path = labels_dir / (img_path.stem + ".txt")
            with Image.open(img_path) as im:
                image = im.convert("RGB")
            boxes = read_yolo_labels(label_path, image.width, image.height)
            if not boxes:
                continue

            for variant in range(variants_per_image):
                # variant마다 ratios를 순서대로 순환 사용 (variants_per_image > len(ratios)면 반복)
                target_ratio = ratios[variant % len(ratios)]
                occluded = image.copy()
                variant_meta = []
                for i, box in enumerate(boxes):
                    occluded, actual_ratio, texture_type = apply_occlusion(
                        occluded, box, target_ratio, rng
                    )
                    variant_meta.append((i, target_ratio, actual_ratio, texture_type))

                out_name = f"{img_path.stem}_occ{int(target_ratio*100)}_v{variant}"
                out_img_path = out_images / f"{out_name}{img_path.suffix}"
                out_lbl_path = out_labels / f"{out_name}.txt"
                occluded.save(out_img_path)

                # 라벨은 원본 full-extent bbox를 그대로 정규화하여 재기록 (occlusion과 무관)
                lines = []
                for cls, x1, y1, x2, y2 in boxes:
                    xc = ((x1 + x2) / 2) / image.width
                    yc = ((y1 + y2) / 2) / image.height
                    w = (x2 - x1) / image.width
                    h = (y2 - y1) / image.height
                    lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                out_lbl_path.write_text("\n".join(lines) + "\n")

                for i, target_ratio_i, actual_ratio, texture_type in variant_meta:
                    writer.writerow([
                        img_path.name, out_img_path.name, i,
                        f"{target_ratio_i:.2f}", f"{actual_ratio:.4f}", texture_type,
                    ])

            print(f"처리 완료: {img_path.name} ({len(boxes)}개 bbox, {variants_per_image}개 variant)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="원본 이미지 디렉토리")
    parser.add_argument("--labels", required=True, help="원본 YOLO 라벨 디렉토리")
    parser.add_argument("--out", required=True, help="출력 디렉토리 (images/, labels/, metadata.csv 생성)")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.3, 0.5, 0.7], help="occlusion 비율 목록")
    parser.add_argument("--variants-per-image", type=int, default=1, help="이미지당 생성할 variant 개수")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    process_dataset(args.images, args.labels, args.out, args.ratios, args.variants_per_image, args.seed)


if __name__ == "__main__":
    main()
