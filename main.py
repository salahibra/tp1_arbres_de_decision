import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Glass Classifier estimator API")

try:
    glass_classifier = joblib.load("glass_classifier.joblib")
    redwine_classifier = joblib.load("redwine_classifier.joblib")
    print("models loaded successfully")

    # with open("glass_classifier.pkl", "rb") as f:
    #     model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError('model file not found')


class PredictionRequestForGlass(BaseModel):
    # ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
    RI: float
    Na: float
    Mg: float
    Al: float
    Si: float
    K: float
    Ca: float
    Ba: float
    Fe: float

@app.post("glass/predict")
def predict(request: PredictionRequestForGlass):
    try:
        features = [[
            request.RI,
            request.Na,
            request.Mg,
            request.Al,
            request.Si,
            request.K,
            request.Ca,
            request.Ba,
            request.Fe,
        ]]
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__== "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)