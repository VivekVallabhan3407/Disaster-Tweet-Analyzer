# 🌍 Disaster Tweet Analyzer

## 🧩 Project Summary
The **Disaster Tweet Analyzer** is an NLP-based system designed to assist in **real-time disaster management** using social media data.  
It analyzes tweets to detect **disaster-related content**, extract **locations and entities**, determine **disaster type**, and assess the **urgency/sentiment** of the situation.  
The project provides an **interactive web dashboard** for visualization and insights.

🔗 **Live Demo:** [Disaster Tweet Analyzer (Streamlit)](https://disaster-tweet-analyzer-yfwtkjjveunmtdqyfntzx6.streamlit.app/)

---

## 🚀 Key Features
1. **Interactive Web-Based Dashboard** (Streamlit / Flask)
2. **Disaster Classification** — Binary (Yes/No) per tweet  
3. **Disaster Type Identification** — via Named Entity Recognition (NER)  
4. **Location Extraction** — identifies places mentioned in tweets  
5. **Purpose & Sentiment/Urgency Analysis**  - Purpose is classified into different catageories and shown. Sentiment of tweets is detected using roberta model and shown as three categories Neutral, Urgent or Positive.
6. **Map-Based Visualization** — shows tweet distribution on a live map  
7. **Deployment on Streamlit Cloud** - Project is deployed live on streamlit cloud.

---

## 🧠 Technology Stack & Requirements

### 🔍 Models and NLP Techniques

| Category | Components | Notes |
| :--- | :--- | :--- |
| **Text Preprocessing** | Tokenization, Lemmatization, Stopword Removal, Lowercasing, Cleaning | For consistent and clean input |
| **Classification Models** | Logistic Regression, Naïve Bayes, BERT (fine-tuned) | Used for disaster detection |
| **NER Models** | spaCy / HuggingFace NER | Extracts disaster type, locations, organizations |
| **Sentiment Analysis** | DistilBERT / Twitter-RoBERTa (Optional) | Determines urgency or sentiment level |
| **Model Storage** | Models are saved and hosted on **Hugging Face Hub** | Enables easy download and deployment |

---

## 📚 Datasets Used
1. **Kaggle:** [NLP Getting Started Dataset](https://www.kaggle.com/competitions/nlp-getting-started/data)  
2. **CrisisNLP - HumAID Dataset (Multi-Class Humanitarian Labels):** [View Dataset](https://crisisnlp.qcri.org/humaid_dataset)  
3. **CrisisNLP - Eyewitness Taxonomy (14k Tweets):** [View Dataset](https://crisisnlp.qcri.org/)

---

---

## 📈 Project Output & Deliverables

1.  **Classification Output:** A clear determination of whether a tweet is disaster-related (`yes`/`no`).
2.  **Entity Output:** A structured list of extracted entities (e.g., `Location: "Miami"`, `Organization: "Red Cross"`).
3.  **Visualization Dashboard:** A web dashboard featuring key statistics, geographical map overlays, and time-series charts of tweet volume.
4.  **Codebase (`src` folder):** Clean, modular, and fully documented Python code for all processing and modeling pipelines.

---


## 📊 Applications
1. **Disaster Management Agencies** → Identify and prioritize high-risk tweets in real time.  
2. **NGOs & Relief Organizations** → Locate areas needing urgent support.  
3. **News Media** → Monitor and report emerging disaster events.  
4. **Research & Academia** → Study crisis informatics and benchmark NLP models.


## ▶️ Getting Started

### Prerequisites

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/VivekVallabhan3407/Disaster-Tweet-Analyzer.git](https://github.com/VivekVallabhan3407/Disaster-Tweet-Analyzer.git)
    cd Disaster-Tweet-Analyzer
    ```
2.  **Setup Environment:** Create and activate a virtual environment.
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The `requirements.txt` file must be created locally first.)*

### Running the App

1.  Place your data files in the `Data/` folder.
2.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run src/app.py
    ```
    *(Assuming your main application file is named `app.py` inside a `src` folder.)*


---

## 🪪 License
© 2025 Vivek Vallabhan. All Rights Reserved.  
This project’s source code and design are proprietary and may not be copied or reproduced.  
Datasets (CrisisNLP and Kaggle) belong to their respective owners and are used only for research and educational purposes.
