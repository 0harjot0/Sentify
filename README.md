# Sentify: Real-time Sentiment Analysis and Insights Platform

Sentify is a backend architecture designed to perform sentiment analysis and provide insights on social media content in real-time, with a focus on platforms like Twitter (via Nitter) and YouTube (Shorts). The system incorporates advanced Natural Language Processing (NLP) and audio processing techniques to analyze text, audio tone, and audio-text content. The android application soruce code - [Link](https://github.com/yoksire/sentimentanalysis)

## Technologies Used:

- **TensorFlow**: An open-source machine learning framework used for building and training deep learning models.
- **Flask**: A micro web framework for building the backend of the application.
- **Hugging Face Transformers**: A library providing pre-trained models for NLP tasks.
- **Librosa**: A Python package for music and audio analysis.
- **PyTorch**: An open-source machine learning library used for various tasks including deep learning.
- **Models**:
  - **Roberta**: Used for text emotion analysis.
  - **Wav2vec2-xlarge**: Speech recognition model.
  - **Hubert-large**: Audio sentiment feature extractor.
  - **Hubert-speech-emotion-recognition**: For emotion recognition in audio.
  - **BILSTM, Dense, CNN**: Base models supporting embedding with Multilingual BERT embeddings and FastText embeddings for sentiment analysis.
  - **DistilBERT Multilingual**: For sentiment analysis on text queries.
  - **Flan-T5**: For rephrasing targeted text posts to increase positive engagement.

## Project Structure:

```
Sentify/
│
├── config/
│   ├── config.yaml
│   └── param.yaml
│
├── src/
│   ├── sentify/
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── data_ingestion.py
│   │   │   ├── data_preparation.py
│   │   │   ├── data_transformation.py
│   │   │   ├── data_validation.py
│   │   │   ├── model_prediction.py
│   │   │   ├── model_trainer.py
│   │   │   └── tweet_scraper.py
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   └── configuration.py
│   │   ├── constants/
│   │   │   └── __init__.py
│   │   ├── entity/
│   │   │   └── __init__.py
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── audio_sentiment.py
│   │   │   ├── classifier.py
│   │   │   ├── emotions.py
│   │   │   ├── prediction.py
│   │   │   ├── scraper.py
│   │   │   ├── training.py
│   │   │   └── youtube_scraper.py
│   │   └── __init__.py
│   ├── logger/
│   │   └── __init__.py
│   └── utility.
│       ├── __init__.py
│       └── common.py
│
├── templates/
│   └── {html files}
│
├── README.md
├── app.py
├── main.py
├── requirements.txt
├── setup.py
└── template.py
```

- **config/**: Contains configuration files `config.yaml` and `param.yaml` for managing project settings.
- **src/**: Source code directory.
  - **sentify/**: Main package directory.
    - **components/**: Submodules for different components of the pipeline such as data ingestion, preparation, transformation, validation, model prediction, and training.
    - **config/**: Configuration module for managing project configurations.
    - **pipeline/**: Pipeline modules for audio sentiment analysis, classification, emotion analysis, prediction, and scraping.
  - **logger/**: Directory for logging functionality.
  - **utility/**: Contains utility scripts and common functions.
- **templates/**: Directory for HTML templates used in the project.
- **README.md**: Project documentation providing an overview and instructions for setup and usage.
- **app.py**: Main application file.
- **main.py**: Entry point for running the application.
- **requirements.txt**: Lists all Python dependencies required to run the project.
- **setup.py**: File for installing the project as a Python package.
- **template.py**: Placeholder file for template generation.

## Setup:

1. Setup Environment:

   ```
   conda create -n venv python=3.10
   conda activate venv
   ```

2. Clone the repository:

   ```
   git clone https://github.com/Subodh7976/Sentify.git
   cd Sentify
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Run the Flask application:

   ```
   python app.py
   ```

5. Access the backend APIs through the defined routes for sentiment analysis, and insights generation.

## Usage:

Once the application is running, the backend APIs can be accessed to perform sentiment analysis, generate insights on social media content, and rephrase text posts for increased positive engagement via Android application.

## Contributors:

- [Subodh Uniyal](https://github.com/Subodh7976)

## License:

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
