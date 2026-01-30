import streamlit as st
import pandas as pd
import re
import requests
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ LOAD NLP ------------------
nlp = spacy.load("en_core_web_sm")

# ------------------ LOAD DATA ------------------
df = pd.read_csv("data_res_com.csv")
ADDRESS_COLUMN = df.columns[0]

def normalize_address(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["normalized_address"] = df[ADDRESS_COLUMN].apply(normalize_address)

# ------------------ LANDMARKS ------------------
LANDMARK_WORDS = {
    "temple","mandir","masjid","school","college","hospital",
    "chai","tapri","bus","station","railway","metro",
    "society","nagar","colony","phase","sector",
    "road","street","rd","st","galli","gaon"
}
POSITIONAL = {"near","opposite","opp","behind","beside","samor","javal"}

def extract_landmarks(address):
    doc = nlp(address)
    return list({
        token.text for token in doc
        if token.text in LANDMARK_WORDS or token.text in POSITIONAL
    })

# ------------------ AUTO CORRECTION ------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_hist = vectorizer.fit_transform(df["normalized_address"])

def autocorrect_address(address):
    vec = vectorizer.transform([address])
    sims = cosine_similarity(vec, X_hist)[0]
    idx = sims.argmax()
    score = float(sims[idx])
    candidate = df["normalized_address"].iloc[idx]

    if score >= 0.75 and len(candidate.split()) >= 4:
        return candidate, round(score,2)
    return address, round(score,2)

# ------------------ INDIA GEO ------------------
def geocode_india(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{address}, India",
        "format": "json",
        "limit": 1,
        "countrycodes": "in",
        "viewbox": "68.7,37.6,97.25,6.75",
        "bounded": 1
    }
    headers = {"User-Agent": "DesiAddressAI"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        data = r.json()
        if not data:
            return None
        return {
            "lat": float(data[0]["lat"]),
            "lon": float(data[0]["lon"]),
            "display_name": data[0]["display_name"]
        }
    except:
        return None

# ------------------ RESOLVE ------------------
def resolve_address(raw):
    normalized = normalize_address(raw)
    landmarks = extract_landmarks(normalized)
    corrected, similarity = autocorrect_address(normalized)
    geo = geocode_india(corrected)

    score = 0.0
    reasons = []

    if len(corrected.split()) >= 6:
        score += 0.3
        reasons.append("Sufficient address detail")

    if len(landmarks) >= 2:
        score += 0.3
        reasons.append("Landmarks detected")

    if similarity >= 0.75:
        score += 0.2
        reasons.append("Historical address similarity high")

    if geo:
        score += 0.2
        reasons.append("India-restricted GPS resolved")

    if score >= 0.7:
        confidence = "HIGH"
    elif score >= 0.4:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "confidence": confidence,
        "confidence_score": round(score,2),
        "confidence_reason": reasons,
        "normalized_address": normalized,
        "corrected_address": corrected,
        "landmarks": landmarks,
        "gps": geo
    }

# ------------------ STREAMLIT UI ------------------
st.title("🚚 DesiAddress AI – Smart Address Intelligence")

address = st.text_area(
    "Enter Indian address",
    "Near chai tapri, opp Hanuman mandir, Andheri East Mumbai"
)

if st.button("Resolve Address"):
    result = resolve_address(address)
    st.json(result)

    if result["gps"]:
        st.map(pd.DataFrame([result["gps"]]))
