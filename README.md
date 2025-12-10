# YouTube Comment Analyzer

This project is a comprehensive tool developed to scrape, analyze, and visualize YouTube comments. It leverages natural language processing (NLP), deep learning-based sentiment analysis, and local Large Language Models (LLM) to provide actionable insights from user feedback.

## Features

- **Multi-Video Analysis:** Scrape and analyze comments from single or multiple videos simultaneously.
- **Battle Mode:** Compare two videos side-by-side to determine which one performs better in terms of user sentiment and engagement.
- **Sentiment Analysis:** Utilizes a BERT-based model (savasy/bert-base-turkish-sentiment-cased) for high-accuracy sentiment classification (Positive, Negative, Neutral).
- **AI-Powered Summaries:** Integrates with local Ollama models (specifically Gemma 2) to generate executive summaries of comments and video descriptions.
- **Advanced Visualizations:** Interactive charts using Plotly, including temporal sentiment analysis, word clouds, and radar charts.
- **Export Capabilities:** Export analyzed data to Excel or CSV formats for further reporting.

## Directory Structure

The project follows a modular architecture:

```
youtube-comment-scraper/
├── app.py                  # Main Streamlit application entry point
├── src/                    # Core source code modules
│   ├── components/         # UI components (charts, word clouds)
│   ├── sentiment_analyzer.py # BERT-based sentiment analysis logic
│   ├── ollama_llm.py       # Wrapper for local LLM integration
│   ├── comment_worker.py   # YouTube scraping logic using yt-dlp
│   └── ...
├── notebooks/              # Jupyter notebooks for data preprocessing and ML experiments
├── data/                   # Directory for storing dataset files
├── requirements.txt        # Python dependency list
└── README.md               # Project documentation
```

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-enabled GPU (recommended for faster sentiment analysis)
- Ollama (for AI summaries)

### Steps

1.  **Clone the Repository**

    ```bash
    git clone <repository-url>
    cd youtube-comment-scraper
    ```

2.  **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Install PyTorch with CUDA Support**
    
    Ensure you install the correct version of PyTorch compatible with your CUDA version. Visit pytorch.org for the specific command.

5.  **Setup Ollama**

    Download and install Ollama from ollama.com. Then pull the required model:

    ```bash
    ollama pull gemma2:latest
    ```

## Usage

1.  **Start the Ollama Server**

    Ensure the Ollama service is running in the background:

    ```bash
    ollama serve
    ```

2.  **Run the Application**

    Launch the Streamlit dashboard:

    ```bash
    streamlit run app.py
    ```

3.  **Analyze Videos**

    - Enter a YouTube video URL to analyze a single video.
    - Switch to "Battle Mode" to compare two videos.
    - Use the search functionality to analyze videos by keyword.

## Tech Stack

- **Frontend:** Streamlit
- **Data Visualization:** Plotly
- **Scraping:** yt-dlp
- **NLP & AI:** Transformers (Hugging Face), Torch, Ollama
- **Data Manipulation:** Pandas, NumPy

## License

This project is open-source and available under the MIT License.
