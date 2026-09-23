from typing import Literal
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

# Load the trained model pipeline
model = joblib.load("/Users/eshwar/Documents/ML Work/Projects/Mental_Health_Score_Predictor/mental_health_pipeline.pkl")

app = FastAPI(title="Mental Health Score Predictor API")

# Enable CORS so frontend (HTML/JS) can call the API without browser blocking
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
top_countries = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']


# ── Request schema ──────────────────────────────────────────────────────────
class Data(BaseModel):
    age                     : int   = Field(..., ge=10, le=100)
    gender                  : Literal["Male", "Female"]
    country                 : str
    academic_level          : Literal["High school", "Undergraduate", "Graduate"]
    most_used_platform      : Literal["Instagram", "Twitter", "Facebook", "Youtube",
                                      "LinkedIn", "LINE", "Snapchat", "Tiktok",
                                      "KakaoTalk", "VKontakte", "Xing", "Whatsapp"]
    purpose_of_use          : Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours   : float = Field(..., ge=0, le=24)
    daily_unlocks           : int   = Field(..., ge=0)
    study_hours             : float = Field(..., ge=0, le=24)
    physical_activity_hours : float = Field(..., ge=0, le=24)
    sleep_hours_per_night   : float = Field(..., ge=0, le=24)
    stress_level            : Literal['Low', 'Medium', 'High', 'Very High']

    @model_validator(mode='after')
    def check_time_feasibility(self):
        sleep    = self.sleep_hours_per_night
        activity = self.physical_activity_hours
        study    = self.study_hours
        usage    = self.avg_daily_usage_hours
        awake    = 24 - sleep

        # Rule 1: sleep + activity are truly exclusive — cannot exceed 24h
        if sleep + activity > 24:
            raise ValueError(
                f"Sleep ({sleep}h) + Physical Activity ({activity}h) = "
                f"{sleep + activity}h, which exceeds 24 hours. "
                "Please check your inputs."
            )

        # Rule 2: screen usage cannot exceed awake hours (24 - sleep)
        if usage > awake:
            raise ValueError(
                f"Daily phone usage ({usage}h) cannot exceed your awake hours "
                f"({awake}h = 24h - {sleep}h sleep). Please adjust sleep or usage."
            )

        # Rule 3: Master Feasibility Check
        # Sleep & Physical Activity require dedicated physical presence.
        # Study and Screen Usage can overlap, so together they take at least max(study, usage).
        overlap_block = max(study, usage)
        total_needed = sleep + activity + overlap_block

        if total_needed > 24:
            dominant = "Study" if study >= usage else "Phone Usage"
            raise ValueError(
                f"Sleep ({sleep}h) + Physical Activity ({activity}h) + {dominant} ({overlap_block}h) = "
                f"{total_needed}h, which exceeds 24 hours in a day. "
                "Please verify your daily schedule inputs."
            )

        return self


# ── Response schema ──────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Welcome to The Mental Health Predictor"}


@app.get("/greet")
def greet():
    return {"message": "Welcome to The Mental Health Predictor"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: Data):
    country_group = data.country if data.country in top_countries else 'Other'

    input_row = pd.DataFrame([{
        'Study_Hours'               : data.study_hours,
        'Age'                       : data.age,
        'Avg_Daily_Usage_Hours'     : data.avg_daily_usage_hours,
        'Daily_Unlocks'             : data.daily_unlocks,
        'Sleep_Hours_Per_Night'     : data.sleep_hours_per_night,
        'Physical_Activity_Hours'   : data.physical_activity_hours,
        'Stress_Level'              : data.stress_level,
        'Gender'                    : data.gender,
        'Academic_Level'            : data.academic_level,
        'Most_Used_Platform'        : data.most_used_platform,
        'Purpose_Of_Use'            : data.purpose_of_use,
        'Grouped_country'           : country_group
    }])

    prediction = model.predict(input_row)[0]
    return PredictionResponse(predicted_mental_health_score=round(float(prediction), 2))
