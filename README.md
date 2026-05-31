# PhysDualNet

Cleaned runnable version for Nino3.4 scalar prediction.

## Contents

- `train.py`: train/test/predict entry point.
- `models/`: `PhysDualNet` and its components.
- `dataset.py`: ENSO dataset loader.
- `PhysDualLoss.py`: PhysDualNet loss.
- `utils.py`: evaluation and plotting utilities.
- `predict_utils.py`: lightweight future-prediction plotting helper.
- `smoke_test.py`: dependency-light model forward test.

## Install

```bash
pip install -r requirements.txt
```


## Train

```bash
python train.py \
  --stage train \
  --model_name PhysDualNet \
  --cmip_path ../processed_ssta_data/cmip6_all_models_processed.nc \
  --obs_val_path ../processed_ssta_data/obs_1958_1978_processed.nc \
  --obs_path ../processed_ssta_data/obs_1980_2021_processed.nc
```

The clean package includes `PhysDualNet` only. The previous references to other model files were removed because those files were not included in the upload.
