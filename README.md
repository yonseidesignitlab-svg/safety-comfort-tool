# Safety-Comfort Tool

A multi-agent tool that scores the **physical safety** (surveillance-facility density)
and **visual comfort** (street-view openness) of subway station areas from public data,
and places each on a four-type **safety–comfort matrix**.

Reference implementation for:
> Youngchae Kim, Jin-Kook Lee, *Development of a Public Data-Based Tool for Evaluating
> the Safety and Comfort of Street Environments — A Case Study of Subway Station Areas
> on Seoul Line 2*, Yonsei University.

---

## ⚙️ What you need to touch — just 2 things

Everything else (code **and** data) is already included.

| Step | What | How |
|------|------|-----|
| 1 | **Add your API key** | `cp .env.example .env`, then paste your Google Street View Static API key into `.env` |
| 2 | **Get the model weight** | `python download_weights.py` (one-time, ~450 MB) |

> The weight file is too large for GitHub (100 MB limit), so it is downloaded once.
> The repository owner sets the download URL in `download_weights.py` (see `models/README.md`).

## ▶️ Run

```bash
pip install -r requirements.txt          # PyTorch: install the build for your platform
cp .env.example .env                      # step 1 — add your key
python download_weights.py                # step 2 — fetch the model

python reproduce/run_line2.py --radius 300        # evaluate the 10 Line-2 stations
python reproduce/run_line2.py --radius 300 --skip-svi   # plan only, no API cost
```

**Result:** `output/station/stations_summary.csv` — per-station `I_phy`, `I_per`, and
matrix quadrant.

### Reproduce the paper figures
```bash
python reproduce/plot_matrix.py          # Fig. 8 — safety–comfort matrix
python reproduce/make_studyarea_map.py   # Fig. 6 — study-area map
```

## 🧩 How it works

An orchestrator runs three agents in sequence:

| Agent | Does | Produces |
|-------|------|----------|
| **Physical Auditor** | counts CCTV / emergency bells / security lights in a station radius, weighted by area | `I_phy` |
| **Perceptual Observer** | samples the road network, fetches Street View, segments it with **DeepLabV3+** | `I_per` |
| **Evaluator** | z-scores both indices and assigns a quadrant | quadrant |

Quadrants (origin = mean of all stations): **Q1 Stable**, **Q2 Facility-Deficient**,
**Q3 Compound-Vulnerable**, **Q4 Enclosure-Dominant**.

## 📁 Layout

```
config.py            parameters & paths (edit weights / radius here; no secrets)
.env.example         template for your API key  → copy to .env
download_weights.py  one-time model download
agents/              orchestrator + physical_auditor + perceptual_observer + evaluator
core/                geo / streetview / io / run-log helpers
DeepLabV3Plus/       vendored segmentation network (MIT, see its LICENSE)
data/                stations + bundled public Seoul datasets (CCTV / bell / light / boundary)
models/              the downloaded weight lands here
reproduce/           run_line2.py + figure scripts
```

## 🔒 Security
No API keys live in the code — they are read from `.env` (git-ignored). Verify `.env` is
never committed; rotate any key that gets exposed.

## 📜 License & attribution
- This project: **MIT** (`LICENSE`).
- Vendored **DeepLabV3+** (segmentation): MIT © 2020 Gongfan Fang —
  https://github.com/VainF/DeepLabV3Plus-Pytorch (`DeepLabV3Plus/LICENSE`).
- Bundled datasets are public Korean open data (Korea Open Government License):
  Seoul Open Data Plaza (CCTV), LocalData (emergency bells), Smart Policing Big Data
  Platform (security lights), administrative-dong boundaries. See `data/README.md`.

If you use this tool, please cite the paper above.
