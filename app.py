from src.sentify.pipeline.training import TrainingPipeline

from flask import Flask, render_template, request
from datetime import date 
import json 
import os 
import time 


PARAMS = {"layers": int, "units": int}

app = Flask(__name__)

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
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
    
