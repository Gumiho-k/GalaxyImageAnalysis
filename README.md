# GalaxyImageAnalysis (refactored skeleton)

This is a cleaned-up project structure based on your original repository. It addresses five immediate issues:

1. normalizes Python file structure and syntax,
2. replaces multiple versioned scripts with one entry point,
3. adds explicit dependencies,
4. adds runnable project documentation,
5. extracts shared logic into reusable modules.

## Project layout

```text
GalaxyImageAnalysis/
├── main.py
├── requirements.txt
├── README.md
└── src/
    └── galaxy_image_analysis/
        ├── __init__.py
        ├── analysis.py
        ├── config.py
        ├── ocr.py
        ├── pipeline.py
        └── segmentation.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need the Tesseract binary installed on your system.

- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`
- Arch: `sudo pacman -S tesseract`

## Run

```bash
python main.py -i _photo.png -o outputs
```

Optional flags:

```bash
python main.py \
  -i _photo.png \
  -o outputs \
  --q0 0.15 \
  --threshold 25 \
  --close-kernel 15 \
  --debug-ocr
```

## Outputs

- `outputs/separated_galaxies/`: cropped galaxy images
- `outputs/deprojected/`: deprojected galaxy images
- `outputs/analysis_plots/`: summary plots for each crop
- `outputs/debug_ocr/`: OCR preprocessing snapshots when `--debug-ocr` is enabled

## Notes

This refactor intentionally keeps the simpler ellipse-based analysis path as the default stable baseline.
Your GPU-heavy 3D fitting scripts can be reintroduced later behind a separate module once the base pipeline is stable and testable.
