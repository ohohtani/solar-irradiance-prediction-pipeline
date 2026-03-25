# GC_final_v2

## Overview

LightGBM-based solar irradiance prediction pipeline for PV plants.
The project includes data preprocessing, midpoint interpolation for selected weather variables, feature engineering, grouped validation by `pv_id`, prediction postprocessing, and visualization export.

## Project Structure

```
gc_final_v2_refactor/
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── modeling.py
│   ├── postprocessing.py
│   └── visualization.py
├── run_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Input Files

Place the following files in the project root:

- `train.csv`
- `test.csv`
- `submission_sample.csv`

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py
```

## Core Pipeline

1. Load train/test/submission files.
2. Optimize numeric dtypes and normalize missing values.
3. Apply midpoint-only filling to `temp_a`, `temp_b`, and `uv_idx` by `pv_id`.
4. Generate time, interaction, location, clustering, and solar geometry features.
5. Split train/validation by `pv_id` using `GroupShuffleSplit`.
6. Train a Tweedie LightGBM regressor.
7. Export validation plots, feature importance, and final submission.

## Output Files

- `GC_final_v2.csv`
- `GC_final_v2_out/model_summary.csv`
- `GC_final_v2_out/feature_importance.csv`
- `GC_final_v2_out/feature_importance.png`
- `GC_final_v2_out/prediction_scatter.png`
- `GC_final_v2_out/location_map.png`
