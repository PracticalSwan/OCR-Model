# Synthetic OCR training data

The OCR upgrade uses synthetic text only as a bounded supplement to the real
public FATURA, FUNSD, and SROIE crops. Large images and font files stay on the
D: asset volume; Git contains the generator, provenance, and aggregate
evidence.

## Corpus composition

The general recognizer generator targets a 20% synthetic share before the
pinned-model dictionary compatibility gate. From the verified real corpus this
produced:

- 119,773 real and 29,943 synthetic general training crops;
- 18,113 real and 4,528 synthetic general validation crops;
- a raw synthetic fraction of 19.9999% in both splits.

The training lists exclude targets that the pinned official dictionary cannot
represent. The usable general lists therefore contain 119,772 real plus 28,072
synthetic training crops and 18,111 real plus 4,245 synthetic DEV_SELECT crops,
for synthetic shares of 18.9876% and 18.9882%, respectively. The gate excludes
2,157 samples rather than silently rewriting their labels: 2,154 synthetic
samples containing U+0E3F and three public real samples containing private-use
code points.

The separate Thai track contains 12,000 synthetic training lines and 2,000
synthetic validation lines. Thai claims must remain limited to this synthetic
development benchmark and integration tests.

The synthetic build ID is `ocr-synthetic-563664fbfa949344`. Its manifest
SHA-256 is
`cbc1fc8950558210df8b684af6bc7f59126673233376ffa12b6d31c1948a4ac4`.
All 48,477 checksum entries were independently replayed successfully.

## Content coverage

General templates cover totals with decimal points and decimal commas, tax and
percentages, invoice and receipt identifiers, dates, references, email
addresses, phone numbers, financial punctuation, major currency symbols, and
Turkish invoice vocabulary and characters.

Thai templates cover store names, addresses, tax identifiers, invoice and
receipt terminology, dates, amounts, Arabic and Thai numerals, baht and THB,
mixed Thai-English names, products, and phone numbers.

Bounded augmentations include mild blur, JPEG compression, brightness and
contrast changes, scanner-like noise, slight shear, small rotations, font-size
variation, padding variation, and background variation. Each sample records
its deterministic seed and exact augmentation parameters. The renderer refuses
single-color or unreadable output.

## Font provenance

The generator downloads fonts from the official
[Google Fonts repository](https://github.com/google/fonts) at pinned commit
`9fab8b6cc7b2f20376914fd765d918c698c66d75`.

- Noto Sans SHA-256:
  `bfb7bb691513f12e734dc346c03a03f784912432d7e3fa8e56efcf906fe86b3d`
- Noto Sans Thai SHA-256:
  `5a1c559bb539583c8a1fd99d1c5b9491e5e14478c9cd2bd0970d5c3096cc9ef8`
- License: SIL Open Font License 1.1, verified from the pinned repository
  [license](https://github.com/google/fonts/blob/9fab8b6cc7b2f20376914fd765d918c698c66d75/ofl/notosansthai/OFL.txt).

Font binaries and licenses remain on D: and are not redistributed by this
repository.

## Thai public-dataset decision

The candidate
[OpenThaiGPT Thai OCR Evaluation dataset](https://huggingface.co/datasets/openthaigpt/thai-ocr-evaluation)
declares CC BY-SA 4.0 and contains 104 test rows. Its card says the images and
text come from various open-source websites, but it does not provide
per-sample source and license mapping. That is not enough to establish the
underlying image redistribution and training rights required here.

The dataset was not downloaded and was not used for training, selection, or
evaluation. The project instead uses the OFL-font synthetic Thai corpus and
does not claim real-world Thai benchmark quality.

## Rebuild

```powershell
python scripts/generate_synthetic_recognition_data.py `
  --real-corpus-root "D:\CSX4201\vision-info-extraction-assets\data\recognition_training" `
  --output-root "D:\CSX4201\vision-info-extraction-assets\data\synthetic_recognition" `
  --font-root "D:\CSX4201\vision-info-extraction-assets\fonts\google-fonts\9fab8b6cc7b2f20376914fd765d918c698c66d75" `
  --synthetic-fraction 0.20 `
  --thai-train-count 12000 `
  --thai-validation-count 2000 `
  --seed 42 `
  --force
```

The generated lists, images, manifests, and checksums are local assets. The
committable aggregate record is
`reports/ocr_upgrade/synthetic_data_manifest.json`.
