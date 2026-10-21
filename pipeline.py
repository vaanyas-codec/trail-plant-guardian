"""
Trail Plant Guardian: inference pipeline (packaged from classifier.ipynb).

    photo -> EfficientNet-B0 top-3 -> confidence tier -> knowledge-graph facts -> advisory

Files expected next to this script: plant_classifier.pth, classes.json, knowledge_graph.pkl
Advisory text comes from a local Ollama model if it is running, otherwise from a
fixed template (so the app never fails and never invents facts).
"""
import json
import pickle

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# --- Config -----------------------------------------------------------------
MODEL_PATH = "plant_classifier.pth"
CLASSES_PATH = "classes.json"
GRAPH_PATH = "knowledge_graph.pkl"
HIGH, LOW = 0.65, 0.30            # tuned on the 14-image val set, applied once to test
WESTERN_GHATS = {"Maharashtra", "Karnataka", "Kerala", "Goa", "Tamil Nadu"}
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Loading ----------------------------------------------------------------
_cache = {}


def load_all():
    """Load model, class list and graph once. Returns (model, classes, G)."""
    if not _cache:
        with open(CLASSES_PATH) as f:
            classes = json.load(f)
        if isinstance(classes, dict):          # tolerate {"0": name} or {name: 0}
            items = list(classes.items())
            classes = ([v for k, v in sorted(items, key=lambda kv: int(kv[0]))]
                       if str(items[0][0]).isdigit()
                       else [k for k, v in sorted(items, key=lambda kv: kv[1])])
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device).eval()
        with open(GRAPH_PATH, "rb") as f:
            G = pickle.load(f)
        _cache.update(model=model, classes=classes, G=G)
    return _cache["model"], _cache["classes"], _cache["G"]


# --- Knowledge-graph retrieval ----------------------------------------------
def recorded_states(G, node):
    return {s: G.edges[node, s]["n_images"]
            for s in G.successors(node)
            if G.edges[node, s]["relation"] == "recorded_in"}


def retrieve(G, plant):
    states = recorded_states(G, plant)
    out = {
        "plant": plant,
        "family": G.nodes[plant].get("family"),
        "plant_states_wg": sorted(s for s in states if s in WESTERN_GHATS),
        "plant_states_other": sorted(s for s in states if s not in WESTERN_GHATS),
        "butterflies": [],
    }
    for b in G.successors(plant):
        if G.edges[plant, b]["relation"] != "hosts":
            continue
        d = G.nodes[b]
        bs = recorded_states(G, b)
        out["butterflies"].append({
            "species": b,
            "common_name": d.get("common_name"),
            "family": d.get("family"),
            "states_wg": sorted(s for s in bs if s in WESTERN_GHATS),
            "has_image_data": d.get("has_metadata", False) and bool(bs),
        })
    return out


def hosts_dict(G):
    """{plant: set(butterflies)} for incident_log.zone_summary."""
    return {n: {b for b in G.successors(n) if G.edges[n, b]["relation"] == "hosts"}
            for n, d in G.nodes(data=True) if d.get("kind") == "plant"}


# --- Classification and gating ----------------------------------------------
def tier_of(conf):
    return "high" if conf >= HIGH else "medium" if conf >= LOW else "low"


def predict_top3(image):
    """`image` may be a path, a file-like object (Streamlit upload) or a PIL image."""
    model, classes, _ = load_all()
    img = image if isinstance(image, Image.Image) else Image.open(image)
    x = eval_tf(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    p, i = probs.topk(3)
    return [(classes[k], float(v)) for v, k in zip(p, i)]


def advise_inputs(image):
    _, _, G = load_all()
    top3 = predict_top3(image)
    tier = tier_of(top3[0][1])
    if tier == "high":
        facts = [retrieve(G, top3[0][0])]
    elif tier == "medium":
        facts = [retrieve(G, n) for n, _ in top3]
    else:
        facts = []
    return {"tier": tier, "facts": facts, "top3": top3}


# --- Advisory text ----------------------------------------------------------
SYSTEM = """You write short trail-side advisories for trekkers in the Western Ghats.
Use ONLY the facts in the JSON you are given. Never add species, places or ecological claims that are not in it.
The one general statement you may make is that caterpillars of the listed butterflies feed on this plant,
so trampling it removes their food.
Say 'recorded as a host' (never 'depends only on'). Use Latin names if a common name is missing.
If a butterfly has no states_wg, don't mention its location. Plain language, no blame, 3-4 sentences."""


def _names(fact, limit=4):
    return [b["common_name"] or b["species"] for b in fact["butterflies"]][:limit]


def template_advisory(fact):
    """Zero-hallucination fallback: every word comes from the retrieved facts."""
    bf = _names(fact)
    if not bf:
        return (f"{fact['plant']} was identified. No host butterflies are recorded for it in "
                "our data, but please stay on the path to protect plants along the trail.")
    more = len(fact["butterflies"]) - len(bf)
    lst = ", ".join(bf) + (f" and {more} more" if more > 0 else "")
    where = ""
    if fact["plant_states_wg"]:
        where = f" It has been recorded in {', '.join(fact['plant_states_wg'])}."
    return (f"{fact['plant']} is recorded as a host plant for butterflies including {lst}."
            f"{where} Their caterpillars feed on this plant, so trampling it removes their food. "
            "Please stay on the marked path.")


def llm_chat(fact, timeout=60):
    r = requests.post(OLLAMA_URL, timeout=timeout, json={
        "model": OLLAMA_MODEL, "stream": False, "options": {"temperature": 0.2},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": json.dumps(fact)}],
    })
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def advisory_for_species(species, use_llm=True):
    """Advisory for a species the user has confirmed. Returns (text, source)."""
    _, _, G = load_all()
    fact = retrieve(G, species)
    if use_llm:
        try:
            return llm_chat(fact), "llm"
        except Exception:
            pass
    return template_advisory(fact), "template"


def grounded(text, fact):
    """Crude leak check: flag any capitalised Latin binomial in `text` that is not in `fact`."""
    import re
    allowed = {fact["plant"]} | {b["species"] for b in fact["butterflies"]}
    found = set(re.findall(r"\b[A-Z][a-z]+ [a-z]{3,}\b", text))
    allowed_words = " ".join(allowed)
    return [f for f in found if f not in allowed_words]
