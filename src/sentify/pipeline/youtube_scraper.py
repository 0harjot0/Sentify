from transformers import pipeline
from huggingsound import SpeechRecognitionModel
from pytube import YouTube 
import moviepy.editor as mp 
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import torchaudio
import torch


class YoutubeSentiment:
    def __init__(self):
        self.speech_model = SpeechRecognitionModel(
            "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        )
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model = HubertForSequenceClassification.from_pretrained("xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
        self.num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad', 4: 'other'}
        
    def extract_captions(self, paths: list[str]):
        transcriptions = self.speech_model.transcribe(paths)
        
        return [response['transcription'] for response in transcriptions]
    
    def audio_sentiment(self, filepath):
        waveform, sample_rate = torchaudio.load(filepath, normalize=True)
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = transform(waveform)

        inputs = self.feature_extractor(
                waveform,
                sampling_rate=self.feature_extractor.sampling_rate,
                return_tensors="pt",
                padding=True,
                max_length=16000 * 10,
                truncation=True
            )

        logits = self.model(inputs['input_values'][0]).logits
        predictions = torch.argmax(logits, dim=-1)
        predicted_emotion = self.num2emotion[predictions.numpy()[0]]
        
        return predicted_emotion
    
    def scrape_url(self, url: str):
        path = YouTube(url).streams.first().download("./artifacts/downloads/")
        
        clip = mp.VideoFileClip(path)
        
        file_path = "./artifacts/downloads/audio.mp3"
        clip.audio.write_audiofile(file_path)
        
        return file_path
        
        