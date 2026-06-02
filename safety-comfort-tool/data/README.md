# Input data

All datasets needed to reproduce the paper are **bundled in this folder** — you do
not need to download them separately.

| File | Content | Source |
|------|---------|--------|
| `line2_stations.json` | the ten study stations (slug, name, lat/lng, district, centrality tier) | this study |
| `seoul_cctv.csv` | public CCTV locations | Seoul Open Data Plaza — Ansimi CCTV linkage |
| `seoul_emergency_bell.csv` | safety emergency-bell locations | LocalData — Ministry of the Interior and Safety |
| `seoul_security_light.csv` | security-light locations | Smart Policing Big Data Platform |
| `seoul_dong.geojson` | Seoul administrative-dong boundaries (filtered from the national 행정동 GeoJSON) | 행정동 경계 GeoJSON |

`config.py` points to these filenames. The facility CSVs use Korean column headers
(`위도` = latitude, `경도` = longitude, `자치구` = district), read directly by
`agents/physical_auditor.py`.

These are public open datasets (Korea Open Government License); they contain
facility coordinates and installation locations only — no personal information.
Redistributed here for reproducibility with attribution to the sources above.
