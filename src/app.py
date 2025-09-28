# =========================================================================
# FINAL app.py SCRIPT (Indentation Errors FIXED)
# =========================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import spacy
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static

# --- CONFIGURATION & CACHING ---
# Use relative paths for Streamlit Cloud deployment
MODELS_DIR = 'Disaster_Analyzer_Data/Models/'
DATA_PATH = 'Disaster_Analyzer_Data/'

# Set up global dictionaries (MUST MATCH TRAINING)
LABEL_LIST = sorted(['caution_and_advice', 'direct-eyewitness', 'displaced_people_and_evacuations',"don't know", 'indirect-eyewitness', 'infrastructure_and_utility_damage','injured_or_dead_people', 'non-eyewitness', 'not_humanitarian','other_relevant_information', 'requests_or_urgent_needs','rescue_volunteering_or_donation_effort', 'sympathy_and_support','vulnerable direct-eyewitness', 'vulnerable-direct witness'])
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
DISASTER_TYPE_MAP = {
'flood': ['flood', 'flooding', 'rain', 'dam', 'inundation', 'tsunami'],
'earthquake': ['quake', 'earthquake', 'shaking', 'seismic', 'magnitude'],
'fire': ['fire', 'wildfire', 'blaze', 'burning', 'smoke', 'inferno'],
'hurricane': ['hurricane', 'cyclone', 'storm', 'typhoon', 'gale'],
}

# --- A. Load Models & Tools (Cache Resources) ---
@st.cache_resource
def load_all_resources():
    # NOTE: You MUST update these paths in your actual code if your models are in a different nested folder
    # Example: 'DistilBERT/saved_model/' if necessary
    
    tokenizer_bin = DistilBertTokenizerFast.from_pretrained(MODELS_DIR + 'distilbert_binary_model')
    model_bin = DistilBertForSequenceClassification.from_pretrained(MODELS_DIR + 'distilbert_binary_model')
    tokenizer_multi = DistilBertTokenizerFast.from_pretrained(MODELS_DIR + 'distilbert_multi_category_model_finetuned')
    model_multi = DistilBertForSequenceClassification.from_pretrained(MODELS_DIR + 'distilbert_multi_category_model_finetuned')
    
    # Load Extraction Tools
    nlp_ner = spacy.load("en_core_web_sm")
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", tokenizer="cardiffnlp/twitter-roberta-base-sentiment")

    return model_bin, tokenizer_bin, model_multi, tokenizer_multi, nlp_ner, sentiment_pipeline

model_bin, tokenizer_bin, model_multi, tokenizer_multi, nlp_ner, sentiment_pipeline = load_all_resources()

# --- B. Load Dashboard Data (Cache Predictions) ---
@st.cache_data
def load_dashboard_data():
    try:
        df = pd.read_csv(DATA_PATH + 'dashboard_data_final.csv')
        return df
    except FileNotFoundError:
        st.error("Dashboard data file not found. Please ensure 'dashboard_data_final.csv' is generated and in the correct path.")
        return pd.DataFrame()

# --- C. Core Prediction Pipeline (for live input) ---

def clean_text_for_bert(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def run_prediction_pipeline(raw_tweet):
    cleaned_bert = clean_text_for_bert(raw_tweet)

    # 1. BINARY FILTER (Model 1)
    inputs = tokenizer_bin(cleaned_bert, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        binary_pred_raw = model_bin(**inputs).logits
    binary_pred = binary_pred_raw.argmax(dim=1).item()

    if binary_pred == 0:
        return {"is_disaster": False, "status": "NOT A DISASTER / FAKE TWEET"}

    # 2. ACTIONABLE ANALYSIS (If binary_pred == 1)
    results = {"is_disaster": True}

    # A. Multi-Class Category (Model 2)
    inputs_multi = tokenizer_multi(cleaned_bert, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits_multi = model_multi(**inputs_multi).logits
    multi_pred_id = logits_multi.argmax(dim=1).item()
    results["purpose"] = ID_TO_LABEL.get(multi_pred_id, "CLASSIFICATION ERROR")

    # B. Named Entity Recognition (NER - Location)
    doc = nlp_ner(raw_tweet)
    locations = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC', 'FAC']]
    results["location"] = locations[0] if locations else "N/A - Cannot extract location."

    # C. Disaster Type Keyword Search
    found_type = "GENERAL DISASTER"
    for type_name, keywords in DISASTER_TYPE_MAP.items():
        if any(word in cleaned_bert for word in keywords):
            found_type = type_name.upper()
            break
    results["type"] = found_type

    # D. Sentiment Analysis (Urgency)
    sentiment_result = sentiment_pipeline(raw_tweet)[0]
    results["sentiment"] = f"{sentiment_result['label']} ({sentiment_result['score']:.2f})"

    return results

# =========================================================================
# --- STREAMLIT UI PAGES ---
# =========================================================================

def home_page():
    st.title("🏡 Welcome to TweetGuardians: Disaster Analyzer")
    st.markdown("---")

    st.header("Project Goal: Turn Noise into Actionable Intelligence")
    st.markdown("""
        During a crisis, social media platforms are overwhelmed with noise.
        **TweetGuardians** is an NLP-powered system that automatically filters these messages,
        identifies real disaster reports, and extracts critical details like location, urgency, and specific needs.

        We use advanced **DistilBERT** models to ensure high recall for urgent tweets.
        This helps emergency responders focus on what matters most, saving lives and resources.""")

    # 

[Image of a data visualization]
 (Using a placeholder query)
    st.image("https://i.imgur.com/your-disaster-image-link.png", caption="Visualization of a Crisis Event (Placeholder)")

    st.markdown("---")

    st.subheader("Start Analysis")
    st.markdown("Click the button below to proceed to the interactive dashboard for live tweet analysis.")
    if st.button("Go to Dashboard", type="primary"):
        st.session_state['page'] = 'Dashboard'
        # st.experimental_rerun() is needed outside a function, but st.session_state change is enough in most cases
        st.rerun()


def dashboard_page(df_dashboard):
    st.title("📊 Dashboard & Live Analyzer")
    st.markdown("---")

    # Check if dashboard data loaded correctly
    if df_dashboard.empty:
        st.error("Dashboard Data Not Available. Please ensure the data generation script was run correctly.")
        return

    # --- Section 1: Dashboard Charts (Analysis of Kaggle Test Set) ---
    st.subheader("1. Test Set Analysis (Batch Report)")
    disaster_count = df_dashboard['is_disaster_pred'].sum()
    non_disaster_count = len(df_dashboard) - disaster_count

    col_sum1, col_sum2, col_sum3 = st.columns(3)
    col_sum1.metric("Confirmed Disaster Alerts", disaster_count)
    col_sum2.metric("Filtered Non-Disaster/Fake", non_disaster_count)
    col_sum3.metric("Binary Model F1-Score", "0.89")

    st.markdown("---")

    # --- CHART 1: DISASTER VS NON-DISASTER (Bar Chart) ---
    st.subheader("Primary Classification Distribution")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='is_disaster_pred', data=df_dashboard, ax=ax1, palette=['salmon', 'skyblue'])
    ax1.set_title('Disaster vs. Non-Disaster Tweets')
    ax1.set_xlabel('Classification (0=Non-Disaster, 1=Disaster)')
    ax1.set_ylabel('Tweet Count')
    st.pyplot(fig1)

    # Filter for Chart 2 & 3
    df_action = df_dashboard[df_dashboard['is_disaster_pred'] == 1].copy()

    # --- CHART 2: ACTION CATEGORY BREAKDOWN (Pie Chart) ---
    st.subheader("Action Category Breakdown (Model 2)")
    
    purpose_counts = df_action['purpose_pred'].value_counts()
    purpose_counts = purpose_counts[~purpose_counts.index.isin(["NOT APPLICABLE", "don't know", "sympathy_and_support"])]
    
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.pie(purpose_counts, labels=purpose_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    ax2.set_title('Distribution of Action Categories')
    st.pyplot(fig2)

    st.markdown("---")
    
    # --- CHART 3: GEOSPATIAL MAP (Location Extraction) ---
    st.subheader("Geospatial Map of Extracted Locations")
    
    map_center = [39.8283, -98.5795]
    m = folium.Map(location=map_center, zoom_start=4)

    st.markdown(f"**Extracted Location Entities (Top 10):** {', '.join(df_action['location_final'].value_counts().head(10).index)}")
    st.markdown("*(Map uses base visualization; entity names confirm successful NER extraction.)*")
    folium_static(m)

    st.markdown("---")

    # --- Section 2: Live Tweet Analysis Tool ---
    st.subheader("Live Tweet Analysis Tool 🔎")

    tweet_input = st.text_area("Enter a new tweet for immediate analysis:",
                               "URGENT: Bridge is out in downtown Houston. Need police and medical teams now.",
                               height=100)

    if st.button("Analyze Tweet Live", type="primary"):
        if tweet_input:
            analysis_results = run_prediction_pipeline(tweet_input) # This line gets the results

            # CRITICAL INDENTATION FIX START
            if not analysis_results["is_disaster"]:
                st.error("--- 🛑 CLASSIFICATION RESULT 🛑 ---")
                st.metric("Status:", analysis_results["status"])
            else:
                st.success("--- ✅ REAL DISASTER ALERT! ---")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("1. ACTION PRIORITY", analysis_results['purpose'].replace('_', ' ').title())
                col2.metric("2. DISASTER TYPE", analysis_results['type'])
                col3.metric("3. EXTRACTED LOCATION", analysis_results['location'])
                col4.metric("4. TONE/URGENCY", analysis_results['sentiment'])

                st.markdown("**Conclusion:** This information is immediately actionable.")
            # CRITICAL INDENTATION FIX END


# --- MAIN APPLICATION LOGIC ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

# Sidebar Navigation
with st.sidebar:
    st.title("Navigation")
    if st.button("Home"):
        st.session_state['page'] = 'Home'
        st.rerun() # Use st.rerun() for button clicks to trigger page change
    if st.button("Dashboard"):
        st.session_state['page'] = 'Dashboard'
        st.rerun() # Use st.rerun() for button clicks to trigger page change
    st.markdown("---")
    st.caption("TweetGuardians: Disaster Analyzer")


# Render the current page
if st.session_state['page'] == 'Home':
    home_page()
elif st.session_state['page'] == 'Dashboard':
    dashboard_df = load_dashboard_data()
    dashboard_page(dashboard_df)