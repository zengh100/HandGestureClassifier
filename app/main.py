from fastapi import FastAPI, File, UploadFile

from ImageClassification.predict_single_image import ImagePredictor

app = FastAPI()

predictor_config_path = "config.yaml"

predictor = ImagePredictor.init_from_config_url(predictor_config_path)


@app.post("/scorefile/")
def create_upload_file(file: UploadFile = File(...)):
    return predictor.predict_from_file(file.file)