from src.sentify.pipeline.training import TrainingPipeline
from src.sentify.pipeline.prediction import PredictionPipeline
from src.sentify.pipeline.scraper import Scraper
from src.sentify.pipeline.audio_sentiment import AudioSentiment
from src.sentify.pipeline.classifier import Classifier
from src.sentify.pipeline.youtube_scraper import YoutubeSentiment

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS 
from datetime import date 
import numpy as np 
import json 
import os 
import time 


SENTIMENT_COUNT = 3
EMOTIONS_COUNT = 11

SENTIMENTS = ['Positive', 'Negative', 'Neutral']
EMOTIONS = ["Joy", "Love", "Optimism", "Pessimism", "Trust", "Surprise", "Anticipation", "Sadness", "Anger", "Disgust", "Fear"]

scraper = Scraper()
predictor = PredictionPipeline()
classifier = Classifier()
youtube_sentiment = YoutubeSentiment()


class MyFlask(Flask):
    def run(self, host=None, port=None, debug=None, 
            load_dotenv=None, **kwargs):
        if not self.debug or os.getenv('WERZEUG_RUN_PATH') == 'true':
            with self.app_context():
                pass
                # global scraper, predictor
                # scraper = Scraper()
                # predictor = PredictionPipeline()
                
        
        super(MyFlask, self).run(host=host, port=port, debug=debug, 
                                 load_dotenv=load_dotenv, **kwargs)

app = MyFlask(__name__)
CORS(app)

PARAMS = {"layers": int, "units": int}

# scraper = None 
# predictor = None 

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html')
    else:
        start_time = time.time()
        
        model_name = request.form.get('model_name')
        embed_type = request.form.get('embed_type')
        
        params = {}
        for param, dtype in PARAMS.items():
            val = request.form.get()
            if val is not None:
                params[param] = dtype(val)
        
        training_pipeline = TrainingPipeline()
        training_pipeline.prepare_pipeline()
        scores = training_pipeline.train_model(model_name, embed_type, **params)
        
        time_taken = time.time() - start_time
        
        return render_template('result.html', 
                               time_taken=time_taken, 
                               scores=scores)
        
@app.route('/scrape', methods=['GET', 'POST'])
def scrape():
    if request.method == "GET":
        return render_template('scrape.html')
    else:
        query = request.form.get("query")
        mode = request.form.get("mode")
        number = int(request.form.get('number'))
        
        response = scraper.scrape_tweets(query, mode, number)

        return render_template("tweets.html", response=response)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict_initial.html')
    else:
        text = request.json['text']
        # texts = ["Hello this is a good example", "this is a bad example", "this is a positive statement", "This is a negative statement"]
        
        emotion_preds = classifier.predict_emotions([text])
        sentiment_preds = classifier.predict_sentiments([text])
        
        print(emotion_preds, sentiment_preds)
        response = {
            "sorted_sentiments": sort_sentiments(sentiment_preds[0]),
            "sorted_emotions": sort_emotions(emotion_preds[0])
        }
        
        return jsonify(response)
    
@app.route('/test', methods=['POST'])
def test():
    query = request.json['query']
    mode = request.json['mode']
    number = int(request.json['number'])
    
    response = scraper.scrape_tweets(query, mode, number)
    # predictions = predictor.predict([tweet['text'] for tweet in response])
    
    preds = predictor.predict([tweet['text'] for tweet in response])
    emotion_preds = preds['Emotions']
    sentiment_preds = preds['Sentiments']
    
    weighted_scores = {'sentiment': [0 for _ in range(SENTIMENT_COUNT)], 
                       'emotions': [0 for _ in range(EMOTIONS_COUNT)]}
    weighted_scores_dict = {'sentiment': {}, 'emotions': {}}
    
    total_likes = 0
    # print(sentiment_preds)
    for i in range(len(response)):
        response[i]['sentiment_prediction'] = sentiment_preds[0][i].tolist()
        response[i]['emotion_prediction'] = emotion_preds[0][i].tolist()
        # weighted_score += response[i]['stats']['likes'] * response[i]['prediction']
        for j in range(SENTIMENT_COUNT):
            weighted_scores['sentiment'][j] += sentiment_preds[0][i].tolist()[j] * response[i]['stats']['likes']
        for j in range(EMOTIONS_COUNT):
            weighted_scores['emotions'][j] += emotion_preds[0][i].tolist()[j] * response[i]['stats']['likes']
        total_likes += response[i]['stats']['likes']
        
        response[i]['sorted_sentiments'] = sort_sentiments1(response[i]['sentiment_prediction'])
        response[i]['sorted_emotions'] = sort_emotions1(response[i]['emotion_prediction'])
                
    if total_likes == 0:
        total_likes = 1
        
    for j in range(SENTIMENT_COUNT):
        weighted_scores['sentiment'][j] /= total_likes
        weighted_scores_dict['sentiment'][SENTIMENTS[j]] = weighted_scores['sentiment'][j] * 100
    for j in range(EMOTIONS_COUNT):
        weighted_scores['emotions'][j] /= total_likes
        weighted_scores_dict['emotions'][EMOTIONS[j]] = weighted_scores['emotions'][j] * 100
        
    final_response = {
        "tweet": response,
        "user": weighted_scores_dict,
    }
    
    with open('response.json', 'w') as file:
        json.dump(final_response, file)
    
    return jsonify(final_response)

@app.route('/audio_test', methods=["POST", "GET"])
def audio_test():
    # processing 
    if request.method == "GET":
        return render_template("audio.html")
    
    f = request.files['file']
    filename = "./artifacts/Audio/main.wav"
    f.save(filename)
    
    audio_sentiment = AudioSentiment()
    scores = audio_sentiment.predict(filename)
    
    average_scores = [0.0, 0.0, 0.0]
    for score in scores:
        average_scores[0] += score[0]
        average_scores[1] += score[1]
        average_scores[2] += score[2]
    
    for i in range(3):
        average_scores[i] /= len(scores)
    
    sentiments = ["Negative", "Neutral", "Positive"]
    
    index = np.argmax(average_scores)
    
    return render_template("audio_result.html", 
                           score=average_scores[index], 
                           sentiment=sentiments[index])
    
@app.route("/video", methods=["GET", "POST"])
def video():
    if request.method == "GET":
        return render_template("video.html")
    
    url = request.json["url"]
    filepath = youtube_sentiment.scrape_url(url)
    audio_sentiment = youtube_sentiment.audio_sentiment(filepath)
    captions = youtube_sentiment.extract_captions([filepath])
    
    emotions = classifier.predict_emotions(captions)
    
    response = {"sentiment": audio_sentiment, "emotions": emotions}
    return jsonify(response)
    
        
def sort_sentiments(inputs):
    # scores = {SENTIMENTS[i]: inputs[i] for i in range(SENTIMENT_COUNT)}
    # sorted_scores = {}
    # for key in  sorted(scores, key=lambda x: scores[x], reverse=True):
    #     sorted_scores[key] = scores[key] * 100
    
    # scores = {SENTIMENTS[i]: inputs[i] for i in range(SENTIMENT_COUNT)}
    scores = {i['label']: i['score'] for i in inputs}
    # sorted_scores = {}
    # for key in  sorted(scores, key=lambda x: scores[x], reverse=True):
    #     sorted_scores[key] = scores[key] * 100
    
    return scores 

def sort_sentiments1(inputs):
    scores = {SENTIMENTS[i]: inputs[i] for i in range(SENTIMENT_COUNT)}
    sorted_scores = {}
    for key in  sorted(scores, key=lambda x: scores[x], reverse=True):
        sorted_scores[key] = scores[key] * 100
    
    return sorted_scores 

def sort_emotions(inputs):
    # scores = {EMOTIONS[i]: inputs[i] for i in range(EMOTIONS_COUNT)}
    # sorted_scores = {}
    # for key in  sorted(scores, key=lambda x: scores[x], reverse=True):
    #     sorted_scores[key] = scores[key] * 100
    
    # return sorted_scores 
    scores = {i['label']: i['score'] for i in inputs}
    
    return scores 

def sort_emotions1(inputs):
    scores = {EMOTIONS[i]: inputs[i] for i in range(EMOTIONS_COUNT)}
    sorted_scores = {}
    for key in  sorted(scores, key=lambda x: scores[x], reverse=True):
        sorted_scores[key] = scores[key] * 100
    
    return sorted_scores 
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
    
