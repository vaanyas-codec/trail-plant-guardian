# Trail Plant Guardian

Confidence-gated host-plant identification and knowledge-graph advisories for butterfly conservation on trekking trails.

A trekker uploads a photo of a plant. The app predicts the species, retrieves the butterflies recorded as using it as a larval host, and lets the user log trampling or damage against a trail zone. Every record keeps the model's prediction, probability and confidence tier next to the species the user confirmed.

> **Status: student prototype (AISA mini-project, MIT World Peace University, Pune).** It is not an autonomous monitoring tool. See [Limitations](#limitations).

<!-- TODO: add a screenshot, e.g. ![High-confidence path](docs/high_confidence.png) -->

## How it works

```
photo -> EfficientNet-B0 top-3 -> confidence gate -> knowledge-graph facts -> advisory -> incident log -> zone dashboard
```

| Top-1 probability | Tier | What happens |
|---|---|---|
| >= 0.65 | High | Top species accepted |
| 0.30 to < 0.65 | Medium | User confirms one of the top 3 (or "none of these") |
| < 0.30 | Low | User retakes the photo or picks from all 14 species |

The advisory defaults to a fixed template filled only from graph facts. An optional local LLM (Ollama, `llama3.2`) can write the text from the same facts; if it is unavailable the app falls back to the template.

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface (report tab, zone dashboard, CSV export) |
| `pipeline.py` | Classification, confidence gating, graph retrieval, advisory text |
| `incident_log.py` | SQLite incident log and zone summaries |
| `plant_classifier.pth` | Fine-tuned EfficientNet-B0 weights (14 classes, about 16 MB) |
| `classes.json` | Class list (order matches the model output) |
| `knowledge_graph.pkl` | NetworkX host-plant graph (14 plants, 39 butterflies, 32 states) |
| `classifier.ipynb` | Training and evaluation |
| `data_cleaning.ipynb` | Dataset cleaning and graph construction |

## Setup

```bash
git clone https://github.com/vaanyas-codec/trail-plant-guardian.git
cd trail-plant-guardian
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

`knowledge_graph.pkl` is a pickle file. Only load pickles you trust.

### Optional: local LLM advisories

Install [Ollama](https://ollama.com), run `ollama pull llama3.2`, keep it running, then switch on **Use local LLM** in the sidebar. This mode has not been systematically evaluated for faithfulness.

## Reproducing the data and training

The notebooks need the raw **IndoLepAtlas** dataset, which is not included here:
<https://huggingface.co/datasets/Butterfree/IndoLepAtlas>

Put the downloaded files where `data_cleaning.ipynb` expects them (check the paths in its first cells), run it, then run `classifier.ipynb`. Training was done once on a CPU with no fixed seed, so exact weights will not be reproduced.

## Results (26 test images)

| Metric | Value (95% Wilson CI) |
|---|---|
| Top-1 accuracy | 61.5% (42.5 to 77.6) |
| Top-3 accuracy | 73.1% (53.9 to 86.3) |
| Top-3 in the medium tier | 10 of 11 images |

Eight of the ten errors lead to an advisory about completely different butterflies, which is why the medium tier requires user confirmation. Full details are in the report and paper.

## Incident data and CSV export

Reports are stored in a local `incidents.db` (SQLite, created on first run, not committed). The **High-impact zones** tab can download incidents and the zone summary as CSV. Rows with `is_demo = 1` are synthetic; remove them before any analysis. Damage is a user observation, not a model output.

## Limitations

- Very small data: 132 images, 26 test images, one training run, no cross-validation.
- Near-duplicate photos across splits were not checked, so accuracy may be inflated.
- Closed set of 14 species. Out-of-distribution photos can reach the medium tier.
- Host links and state records come from dataset metadata and were not expert-reviewed. Whole states stand in for the Western Ghats.
- Trail zones and coordinates in `incident_log.py` are **placeholders**. The Watch (>= 2) and High (>= 5) thresholds are arbitrary defaults.
- No study has been run with trekkers or researchers. Nothing here shows reduced trampling or benefit to butterflies.

## Planned, not built

GBIF / iNaturalist integration, surveyed zones and coordinates, expert review of host links, an "unknown" class with probability calibration, and image-based damage detection.

## Data and licence

Plant and butterfly data come from IndoLepAtlas (CC BY-NC-SA 4.0), whose photographs are credited to their photographers and to The Butterflies of India Consortium on iFoundButterflies. Because of the NonCommercial and ShareAlike terms, check how they apply to the trained weights and graph before reuse or redistribution.

<!-- TODO: add a LICENSE file for your own code (e.g. MIT) and confirm it is compatible with the data licence. -->

## Authors

Vaanya Kshatriya and Vansh Hardwani. Faculty guide: Dr. Shital Girase.
