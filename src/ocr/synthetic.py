"""Deterministic public-safe synthetic OCR line generation."""
from __future__ import annotations

import io
import math
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


GENERAL_CATEGORIES = (
    "amount",
    "amount_european",
    "tax",
    "invoice_identifier",
    "receipt_identifier",
    "date_slash",
    "date_iso",
    "thai_baht_amount",
    "currency_amount",
    "tax_identifier",
    "email",
    "phone",
    "reference",
    "turkish_invoice",
    "turkish_tax",
    "percentage",
)
THAI_CATEGORIES = (
    "store",
    "address",
    "tax_identifier",
    "invoice",
    "date_arabic",
    "date_thai_digits",
    "amount",
    "currency",
    "mixed_name",
    "product",
    "phone",
    "receipt",
)


def synthetic_count_for_fraction(real_count: int, fraction: float) -> int:
    """Return synthetic count when fraction refers to the combined corpus."""
    if real_count <= 0:
        raise ValueError("real_count must be positive")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between zero and one")
    return int(round(real_count * fraction / (1.0 - fraction)))


def general_financial_text(index: int, rng: random.Random) -> str:
    category = GENERAL_CATEGORIES[index % len(GENERAL_CATEGORIES)]
    amount = rng.randint(1, 9999) + rng.randint(0, 99) / 100
    number = f"{amount:,.2f}"
    european = number.replace(",", "_").replace(".", ",").replace("_", ".")
    date = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2020, 2028)}"
    templates = {
        "amount": f"TOTAL {number}",
        "amount_european": f"TOTAL {european} €",
        "tax": f"VAT {rng.choice((7, 10, 18, 20))}% TAX {number}",
        "invoice_identifier": f"INV-{rng.randint(2020, 2028)}-{rng.randint(1, 99999):05d}",
        "receipt_identifier": f"Receipt No. R-{rng.randint(1, 999999):06d}",
        "date_slash": date,
        "date_iso": f"{rng.randint(2020, 2028)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "thai_baht_amount": f"฿{number}",
        "currency_amount": f"{rng.choice(('THB', 'USD', 'GBP', 'JPY'))} {number}",
        "tax_identifier": f"Tax ID {rng.randint(10**12, 10**13 - 1)}",
        "email": f"billing{rng.randint(1, 999)}@example.org",
        "phone": f"+66 2 {rng.randint(100, 999)} {rng.randint(1000, 9999)}",
        "reference": f"REF#{rng.randint(100000, 999999)} / AC-{rng.randint(1000, 9999)}",
        "turkish_invoice": f"FATURA Çeşitli Ürünler No: TR-{rng.randint(1000, 9999)}",
        "turkish_tax": f"TOPLAM {european} ₺ KDV Ödeme İrsaliye",
        "percentage": f"DISCOUNT ({rng.randint(1, 50)}%) - TAX: {number}",
    }
    return templates[category]


def thai_financial_text(index: int, rng: random.Random) -> str:
    category = THAI_CATEGORIES[index % len(THAI_CATEGORIES)]
    amount = f"{rng.randint(1, 99999):,}.{rng.randint(0, 99):02d}"
    thai_digits = str.maketrans("0123456789", "๐๑๒๓๔๕๖๗๘๙")
    arabic_date = (
        f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2566, 2572)}"
    )
    templates = {
        "store": rng.choice(
            ("ร้านสยามมาร์ท", "บริษัท กรุงเทพการค้า จำกัด", "ห้างไทยรุ่งเรือง")
        ),
        "address": f"ที่อยู่ {rng.randint(1, 999)} ถนนสุขุมวิท กรุงเทพฯ 10110",
        "tax_identifier": f"เลขประจำตัวผู้เสียภาษี {rng.randint(10**12, 10**13 - 1)}",
        "invoice": f"ใบกำกับภาษี เลขที่ INV-{rng.randint(2020, 2028)}-{rng.randint(1, 99999):05d}",
        "date_arabic": f"วันที่ {arabic_date}",
        "date_thai_digits": f"วันที่ {arabic_date.translate(thai_digits)}",
        "amount": f"รวมทั้งสิ้น ฿{amount} บาท",
        "currency": f"ยอดชำระ THB {amount}",
        "mixed_name": rng.choice(
            ("คุณ Somchai ใจดี", "บริษัท Siam Digital จำกัด", "ร้าน ABC เชียงใหม่")
        ),
        "product": rng.choice(
            ("กาแฟเย็น 2 แก้ว", "ข้าวหอมมะลิ 5 กก.", "บริการจัดส่ง Express")
        ),
        "phone": f"โทร {rng.randint(80, 99):02d}-{rng.randint(100, 999):03d}-{rng.randint(1000, 9999):04d}",
        "receipt": f"ใบเสร็จรับเงิน เลขที่ R-{rng.randint(1, 999999):06d}",
    }
    return templates[category]


def render_synthetic_line(
    text: str,
    *,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    seed: int,
) -> tuple[Image.Image, dict[str, Any]]:
    """Render one legible line with bounded scanner-like augmentation."""
    if not text.strip() or any(character in text for character in "\t\r\n"):
        raise ValueError("synthetic transcription must be one nonempty line")
    rng = random.Random(seed)
    numpy_rng = np.random.default_rng(seed)
    probe = Image.new("L", (16, 16), 255)
    probe_draw = ImageDraw.Draw(probe)
    bbox = probe_draw.textbbox((0, 0), text, font=font)
    text_width = max(1, bbox[2] - bbox[0])
    text_height = max(1, bbox[3] - bbox[1])
    horizontal_padding = rng.randint(12, 30)
    vertical_padding = rng.randint(8, 18)
    width = max(32, text_width + horizontal_padding * 2)
    height = max(16, text_height + vertical_padding * 2)
    background_level = rng.randint(235, 255)
    noise_sigma = rng.uniform(0.0, 4.5)
    pixels = numpy_rng.normal(
        loc=background_level,
        scale=noise_sigma,
        size=(height, width),
    )
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(image)
    foreground = rng.randint(0, 55)
    x = horizontal_padding + rng.randint(-2, 2)
    y = vertical_padding - bbox[1] + rng.randint(-1, 1)
    draw.text(
        (x, y),
        text,
        fill=foreground,
        font=font,
        stroke_width=1 if rng.random() < 0.08 else 0,
    )

    blur_radius = rng.uniform(0.0, 0.75)
    if blur_radius > 0.15:
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    brightness = rng.uniform(0.88, 1.10)
    contrast = rng.uniform(0.82, 1.18)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    shear = rng.uniform(-0.018, 0.018)
    if abs(shear) > 0.003:
        image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1.0, shear, -shear * image.height / 2.0, 0.0, 1.0, 0.0),
            resample=Image.Resampling.BICUBIC,
            fillcolor=background_level,
        )
    rotation = rng.uniform(-2.0, 2.0)
    if abs(rotation) > 0.15:
        image = image.rotate(
            rotation,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=background_level,
        )
    jpeg_quality = rng.randint(62, 96)
    if rng.random() < 0.75:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            compressed.load()
            image = compressed.convert("L")
    extrema = image.getextrema()
    if extrema is None or extrema[1] - extrema[0] < 20:
        raise ValueError("synthetic line became unreadable")
    metadata = {
        "background_level": background_level,
        "noise_sigma": round(noise_sigma, 6),
        "blur_radius": round(blur_radius, 6),
        "brightness": round(brightness, 6),
        "contrast": round(contrast, 6),
        "shear": round(shear, 6),
        "rotation_degrees": round(rotation, 6),
        "jpeg_quality": jpeg_quality,
    }
    return image, metadata
