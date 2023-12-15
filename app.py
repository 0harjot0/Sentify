from src.sentify.pipeline.training import TrainingPipeline
from src.sentify.pipeline.scraper import Scraper

from flask import Flask, render_template, request
from datetime import date 
import json 
import os 
import time 


class MyFlask(Flask):
    def run(self, host=None, port=None, debug=None, 
            load_dotenv=None, **kwargs):
        if not self.debug or os.getenv('WERZEUG_RUN_PATH') == 'true':
            with self.app_context():
                global scraper
                scraper = Scraper()
        
        super(MyFlask, self).run(host=host, port=port, debug=debug, 
                                 load_dotenv=load_dotenv, **kwargs)

app = MyFlask(__name__)

PARAMS = {"layers": int, "units": int}

scraper = None 

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
        
@app.route('/scrape')
def scrape():
    if request.method == "GET":
        return render_template('scrape.html')
    else:
        query = request.form.get("query")
        mode = request.form.get("mode")
        number = int(request.form.get('number'))
        
        response = scraper.scrape_tweets(query, mode, number)

        return render_template("tweets.html", response=response)
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
    
