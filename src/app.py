import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim

# --- CONFIGURATION & CACHING ---
DATA_PATH = 'Data/'

# Hugging Face Hub IDs
HF_USER = "Vivek1564"
BIN_REPO_ID = f"{HF_USER}/disaster-tweet-binary-filter"
MULTI_REPO_ID = f"{HF_USER}/disaster-tweet-multi-category-finetuned"

# Label Mappings
LABEL_LIST = sorted([
    'caution_and_advice', 'direct-eyewitness', 'displaced_people_and_evacuations',
    "don't know", 'indirect-eyewitness', 'infrastructure_and_utility_damage',
    'injured_or_dead_people', 'non-eyewitness', 'not_humanitarian',
    'other_relevant_information', 'requests_or_urgent_needs',
    'rescue_volunteering_or_donation_effort', 'sympathy_and_support',
    'vulnerable direct-eyewitness', 'vulnerable-direct witness'
])
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

# Disaster type keywords
DISASTER_TYPE_MAP = {
    'flood': ['flood', 'flooding', 'rain', 'dam', 'inundation', 'tsunami'],
    'earthquake': ['quake', 'earthquake', 'shaking', 'seismic', 'magnitude'],
    'fire': ['fire', 'wildfire', 'blaze', 'burning', 'smoke', 'inferno'],
    'hurricane': ['hurricane', 'cyclone', 'storm', 'typhoon', 'gale'],
    'volcano': ['volcano', 'eruption', 'lava', 'ash', 'pyroclastic'],
    'landslide': ['landslide', 'mudslide', 'rockslide', 'avalanche'],
    'pandemic': ['covid', 'pandemic', 'virus', 'disease', 'epidemic', 'outbreak'],
    'tornado': ['tornado', 'twister', 'funnel cloud'],
    'drought': ['drought', 'dry spell', 'heatwave', 'famine'],
    'explosion': ['explosion', 'blast', 'bomb', 'detonation', 'attack']
}
DEFAULT_MAP_CENTER = (39.8283, -98.5795)

# Sentiment Mapping
SENTIMENT_MAP = {
    'LABEL_0': 'NEGATIVE (Distress)',
    'LABEL_1': 'NEUTRAL (Factual/Urgent)',
    'LABEL_2': 'POSITIVE (Support)'
}

# --- A. Load Models & Tools ---
@st.cache_resource
def load_all_resources():
    # Load from Hugging Face Hub
    tokenizer_bin = DistilBertTokenizerFast.from_pretrained(BIN_REPO_ID)
    model_bin = DistilBertForSequenceClassification.from_pretrained(BIN_REPO_ID)

    tokenizer_multi = DistilBertTokenizerFast.from_pretrained(MULTI_REPO_ID)
    model_multi = DistilBertForSequenceClassification.from_pretrained(MULTI_REPO_ID)

    nlp_ner = spacy.load("en_core_web_sm")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment"
    )

    return model_bin, tokenizer_bin, model_multi, tokenizer_multi, nlp_ner, sentiment_pipeline

model_bin, tokenizer_bin, model_multi, tokenizer_multi, nlp_ner, sentiment_pipeline = load_all_resources()

# --- B. Load Dashboard Data ---
def clean_text_for_bert(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def infer_disaster_type(text):
    text = str(text).lower()
    for type_name, keywords in DISASTER_TYPE_MAP.items():
        if any(word in text for word in keywords):
            return type_name.upper()
    return "GENERAL DISASTER"

@st.cache_data
def load_dashboard_data():
    try:
        df = pd.read_csv(DATA_PATH + 'dashboard_data_final.csv')
        if 'text_cleaned' not in df.columns and 'text' in df.columns:
            df['text_cleaned'] = df['text'].apply(clean_text_for_bert)
        if 'disaster_type_pred' not in df.columns:
            df['disaster_type_pred'] = df['text_cleaned'].apply(infer_disaster_type)
        return df
    except FileNotFoundError:
        st.error("Dashboard data file not found.")
        return pd.DataFrame()

# --- C. Core Prediction Pipeline ---
def run_prediction_pipeline(raw_tweet):
    cleaned_bert = clean_text_for_bert(raw_tweet)

    # Binary classification
    inputs = tokenizer_bin(cleaned_bert, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        binary_pred_raw = model_bin(**inputs).logits
    binary_pred = binary_pred_raw.argmax(dim=1).item()

    if binary_pred == 0:
        return {"is_disaster": False, "status": "NOT A DISASTER / FAKE TWEET"}

    results = {"is_disaster": True}

    # Multi-class prediction
    inputs_multi = tokenizer_multi(cleaned_bert, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits_multi = model_multi(**inputs_multi).logits
    multi_pred_id = logits_multi.argmax(dim=1).item()
    results["purpose"] = ID_TO_LABEL.get(multi_pred_id, "CLASSIFICATION ERROR")

    # Named entity recognition for location
    doc = nlp_ner(raw_tweet)
    locations = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC', 'FAC']]
    results["location"] = locations[0] if locations else "N/A"

    # Geocoding
    geolocator = Nominatim(user_agent="disaster-tweet-analyzer-app")
    lat_lon = None
    if results["location"] != "N/A":
        try:
            location = geolocator.geocode(results["location"], timeout=5)
            if location:
                lat_lon = (location.latitude, location.longitude)
        except:
            pass
    results["coords"] = lat_lon if lat_lon else DEFAULT_MAP_CENTER
    results["type"] = infer_disaster_type(cleaned_bert)

    # Sentiment
    sentiment_result = sentiment_pipeline(raw_tweet)[0]
    mapped_label = SENTIMENT_MAP.get(sentiment_result['label'], sentiment_result['label'])
    results["sentiment"] = f"{mapped_label} ({sentiment_result['score']:.2f})"

    return results

# --- STREAMLIT UI PAGES ---
def home_page():
    st.title("🏡 Welcome to TweetGuardians")
    st.markdown("---")
    st.markdown(
        """
        <div style="background-color:#f0f4f8;padding:20px;border-radius:10px">
        <h3>Turn Noise into Actionable Intelligence</h3>
        <p>During a crisis, social media is flooded with noise. TweetGuardians filters messages, identifies real disaster reports, 
        and extracts critical details like location, urgency, and humanitarian purpose.</p>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("---")
    if st.button("🚀 Go to Dashboard", type="primary"):
        st.session_state['page'] = 'Dashboard'
        st.rerun()

def dashboard_page(df_dashboard):
    st.title("📊 Disaster Dashboard & Live Analyzer")
    st.markdown("---")

    if df_dashboard.empty:
        st.error("Dashboard Data Not Available.")
        return

    # Metrics
    disaster_count = df_dashboard['is_disaster_pred'].sum()
    non_disaster_count = len(df_dashboard) - disaster_count
    col1, col2, col3 = st.columns(3)
    col1.metric("🌪️ Confirmed Alerts", f"{disaster_count}", delta_color="off")
    col2.metric("🚫 Filtered Non-Disaster", f"{non_disaster_count}", delta_color="off")
    col3.metric("📈 Binary Model F1-Score", "0.89", delta_color="off")
    st.markdown("---")

    # Charts
    df_action = df_dashboard[df_dashboard['is_disaster_pred'] == 1].copy()

    # Humanitarian Action Breakdown
    if 'purpose_pred' in df_action.columns:
        purpose_counts = df_action['purpose_pred'].value_counts()
        purpose_counts = purpose_counts[~purpose_counts.index.isin(["NOT APPLICABLE", "don't know", "sympathy_and_support"])]
        fig, ax = plt.subplots(figsize=(22,14))
        sns.barplot(x=purpose_counts.values, y=purpose_counts.index, palette="Set2", ax=ax)
        ax.set_title("Humanitarian Action Breakdown", fontsize=24, weight='bold')
        ax.set_xlabel("Number of Tweets", fontsize=20)
        ax.set_ylabel("Action Category", fontsize=20)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        st.pyplot(fig)

    st.markdown("---")

    # Disaster Type Breakdown
    if 'disaster_type_pred' in df_action.columns:
        type_counts = df_action['disaster_type_pred'].value_counts()
        fig, ax = plt.subplots(figsize=(22,14))
        sns.barplot(x=type_counts.values, y=type_counts.index, palette="Set3", ax=ax)
        ax.set_title("Disaster Type Breakdown", fontsize=24, weight='bold')
        ax.set_xlabel("Number of Tweets", fontsize=20)
        ax.set_ylabel("Disaster Type", fontsize=20)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        st.pyplot(fig)

    st.markdown("---")

    # Live Tweet Analysis
    st.subheader("🔎 Live Tweet Analysis")
    tweet_input = st.text_area("Enter a tweet:", "URGENT: Earthquake in downtown Mumbai. Need police and medical teams now.", height=150)

    if st.button("Analyze Tweet Live", type="primary"):
        results = run_prediction_pipeline(tweet_input)
        if not results["is_disaster"]:
            st.error("🛑 Not a Disaster Tweet")
            st.info(f"Status: {results['status']}")
        else:
            st.success("✅ Real Disaster Alert")

            # Info Cards
            card_col1, card_col2, card_col3, card_col4 = st.columns(4)
            card_col1.markdown(f"<div style='border:2px solid #ff4b4b;padding:20px;border-radius:10px;text-align:center;font-size:18px'><b>Action:</b><br>{results['purpose'].replace('_',' ').title()}</div>", unsafe_allow_html=True)
            card_col2.markdown(f"<div style='border:2px solid #ffa500;padding:20px;border-radius:10px;text-align:center;font-size:18px'><b>Type:</b><br>{results['type']}</div>", unsafe_allow_html=True)
            card_col3.markdown(f"<div style='border:2px solid #1f77b4;padding:20px;border-radius:10px;text-align:center;font-size:18px'><b>Location:</b><br>{results['location']}</div>", unsafe_allow_html=True)
            card_col4.markdown(f"<div style='border:2px solid #2ca02c;padding:20px;border-radius:10px;text-align:center;font-size:18px'><b>Tone/Urgency:</b><br>{results['sentiment']}</div>", unsafe_allow_html=True)

            # Map
            st.subheader("🗺️ Detected Location on Map")
            map_center = results["coords"]
            m = folium.Map(location=map_center, zoom_start=7)
            folium.Marker(map_center, tooltip=results["location"], popup=f"{results['type']}").add_to(m)
            folium_static(m)

# --- MAIN APP LOGIC ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

with st.sidebar:
    st.title("Navigation")
    if st.button("Home"):
        st.session_state['page'] = 'Home'
        st.rerun()
    if st.button("Dashboard"):
        st.session_state['page'] = 'Dashboard'
        st.rerun()
    st.markdown("---")
    st.caption("TweetGuardians: Disaster Analyzer")

if st.session_state['page'] == 'Home':
    home_page()
elif st.session_state['page'] == 'Dashboard':
    dashboard_df = load_dashboard_data()
    dashboard_page(dashboard_df)
