# Disaster Tweet Analyzer

## Project Summary

The **Disaster Tweet Analyzer** is a natural language processing (NLP) project focused on leveraging social media data to assist in real-time disaster management. The primary objective is to swiftly detect crisis-related tweets, accurately extract critical information such as **locations** and **organizations**, and provide actionable insights into the urgency and sentiment of the event.

The tool is designed for efficiency and is built to handle bulk CSV data processing as well as live input from Twitter streams.

---

## 🚀 Key Features

* **Disaster Classification:** High-accuracy binary classification (Disaster vs. Non-Disaster) for filtering noisy social data.
* **Entity Extraction (NER):** Detects and extracts critical entities, including **locations** and **organizations**, vital for response coordination.
* **Interactive Dashboard:** Provides a complete overview of the analysis, including statistics, charts, and a map visualization of extracted locations.
* **Bulk & Live Processing:** Supports uploading large CSV datasets and handles real-time data input.
* **Optional Sentiment Analysis:** (Planned/Future Feature) Provides a layer of urgency/sentiment analysis (e.g., positive, negative, neutral) to prioritize response efforts.

---

## 🛠️ Technology Stack & Requirements

### Models and NLP Techniques

| Category | Components | Notes |
| :--- | :--- | :--- |
| **Techniques** | Text Cleaning (URLs, Mentions, Punctuation), Tokenization, Stopword Removal, Lemmatization, Lowercasing. | Standard text preparation pipeline. |
| **Baseline Models** | Logistic Regression, Naïve Bayes | Used for fast, interpretable baselines to measure complex model lift. |
| **Core Models** | **BERT** (Fine-tuned for classification), **spaCy / HuggingFace NER** | Focus on achieving state-of-the-art accuracy and entity extraction precision. |
| **Optional/Stretch** | Twitter-RoBERTa, DistilBERT SST-2 | For advanced sentiment and urgency analysis. |

### Data Sources

The project utilizes publicly available, verified disaster-related tweet datasets:

* **Primary:** [Kaggle NLP Getting Started Competition Data](https://www.kaggle.com/competitions/nlp-getting-started/data)
* **Secondary (Optional):** Selected datasets from the **CrisisNLP** corpus (e.g., specific earthquakes, floods, or hurricanes).

### Deployment

The final interactive application will be deployed using:

* **Streamlit** for the web application interface.
* **Streamlit Cloud** or **HuggingFace Spaces** for deployment.

---

## 📈 Project Output & Deliverables

1.  **Classification Output:** A clear determination of whether a tweet is disaster-related (`yes`/`no`).
2.  **Entity Output:** A structured list of extracted entities (e.g., `Location: "Miami"`, `Organization: "Red Cross"`).
3.  **Visualization Dashboard:** A web dashboard featuring key statistics, geographical map overlays, and time-series charts of tweet volume.
4.  **Codebase (`src` folder):** Clean, modular, and fully documented Python code for all processing and modeling pipelines.

---

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