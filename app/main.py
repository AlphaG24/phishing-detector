from fastapi import FastAPI
import joblib
from pydantic import BaseModel

# Create the FastAPI app
app = FastAPI()

# Load the trained model and vectorizer
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Define the request body structure
class EmailRequest(BaseModel):
    email_text: str

# Define the prediction endpoint
@app.post("/predict")
def predict(request: EmailRequest):
    # The user's text is in request.email_text
    text = [request.email_text]

    # 1. Vectorize the input text
    vectorized_text = vectorizer.transform(text)

    # 2. Make a prediction
    prediction = model.predict(vectorized_text)
    prediction_label = prediction[0]

    # 3. Get the prediction probability
    probability = model.predict_proba(vectorized_text).max()

    # Return the result
    return {
        "prediction": prediction_label,
        "confidence_score": f"{probability*100:.2f}%"
    }

# Define a root endpoint for simple testing
@app.get("/")
def read_root():
    return {"message": "Phishing Email Detector API is running!"}