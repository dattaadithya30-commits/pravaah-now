# References & Datasets

Matches the citations on our PPT's "Research and References" slide.

## Datasets

| Source | Link | Access |
|---|---|---|
| MOSDAC | https://mosdac.gov.in | **Have access** |
| NCMRWF IMDAA | https://rds.ncmrwf.gov.in | Open (reanalysis — not live; used for offline validation only) |
| BharatBench | https://kaggle.com/datasets/maslab/bharatbench | Train-only |
| IMD Mausam | https://mausam.imd.gov.in | Open |

## Papers

1. **arXiv:2607.16080** — ConvLSTM spatiotemporal sequence modeling for radar nowcasting.
2. **IEEE JSTARS** (doi:10.1109/JSTARS.2026.3729834) — Physics-guided U-Net fusion for extreme rainfall estimation.
3. **SSRN:6504081** — Multi-modal satellite and radar inputs for convective forecasting.
4. **arXiv:2601.06137** — Intensity-weighted loss formulation preserving severe storm cores (basis for `model/losses.py`).

## Note on data pathway

IMDAA is reanalysis, not a live feed. Our live operational pathway is:
**MOSDAC + IMD DWR + lightning + AWS** — extending the IMD-HRRR and
IMDAA architecture rather than depending on IMDAA directly at inference
time.
