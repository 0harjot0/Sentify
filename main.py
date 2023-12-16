from sentify.pipeline.training import TrainingPipeline

import sys 

params = {
    "layers": 2,
    'units': 64
}

args = sys.argv

if args is None:
    print("Define Model argument")
    exit()

embed_name, model_name = args[-1].split('_')

training_pipeline = TrainingPipeline()

print("pipeline started")
# training_pipeline.prepare_pipeline()

print("Pipeline Prepared")
if model_name == "cnn":
    params = {"units": params['units']}
training_pipeline.train_model(model_name, embed_name, **params)
print("Training Completed")
