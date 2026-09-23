# 🧠 Student Mental Health Score Predictor

A machine learning project that predicts a student's mental health score (on a 1–10 scale) based on their social media usage, sleep habits, physical activity, and study routines.

It includes:
- A trained machine learning model (**Random Forest**) built with scikit-learn.
- A **FastAPI** backend providing the prediction API.
- A clean, interactive web interface (`index.html`) to test inputs and view predictions live.
- Built-in time validation to prevent impossible inputs (e.g., days exceeding 24 hours).

---

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn, Pydantic
- **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Exploration & Training:** Jupyter Notebook (`ML_Project (1).ipynb`)

---

## ⚙️ How It Works

1. **Input Data:** The user fills in their daily habits (sleep, study hours, social media usage, physical activity, phone unlocks, stress level, etc.).
2. **Time Validation:** The app checks that the hours make physical sense before sending anything to the model.
3. **Prediction:** The backend loads the trained model pipeline (`mental_health_pipeline.pkl`), preprocesses the inputs, and calculates the predicted score.
4. **Results:** The frontend displays the predicted score on an animated gauge with a quick summary.

---

## ⏱️ The 24-Hour Day Validation Rule

A common issue with habit-based models is that users can enter conflicting numbers without realizing it:
* For example, entering **20 hours of sleep**, **1 hour of workout**, and **4 hours of phone usage**.
* That adds up to **25 hours**, which is physically impossible in a 24-hour day.

Rather than letting the model output a guess on impossible numbers, the app validates inputs on both the frontend (live warning) and backend (FastAPI 422 error):

1. **Sleep + Physical Activity cannot exceed 24h** (you cannot exercise while asleep).
2. **Phone usage cannot exceed awake time** (`24 - sleep`).
3. **Total committed time must fit in a day:**
   $$\text{Sleep} + \text{Physical Activity} + \max(\text{Study Hours}, \text{Phone Usage}) \le 24$$
   *(Since study time and phone usage can partially overlap, we account for whichever is larger).*

If an input exceeds 24 hours, the UI displays a clear red warning box and disables submission until the values are adjusted.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Eshwaranand26/Mental_Health_Score_Predictor.git
cd Mental_Health_Score_Predictor
```

### 2. Create and activate a virtual environment
```bash
# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the application
```bash
uvicorn main:app --reload --port 8000
```

### 5. Open in browser
- **Web App:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive API Docs (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📊 Machine Learning Pipeline

The complete training and analysis workflow is documented in [`ML_Project (1).ipynb`](./ML_Project%20(1).ipynb):

- **Dataset:** 5,000 student survey records with demographic, academic, and behavioral features.
- **Preprocessing:**
  - Scaled numerical features (`Study_Hours`, `Daily_Unlocks`, `Sleep_Hours_Per_Night`, etc.) using `StandardScaler`.
  - Encoded nominal features (`Gender`, `Most_Used_Platform`, `Purpose_Of_Use`, grouped `Country`) using `OneHotEncoder`.
  - Mapped ordinal features (`Stress_Level`, `Academic_Level`) to ordered numeric scales.
- **Model Evaluation:** Tested Linear Regression as a baseline and Random Forest Regressor. The tuned Random Forest model yielded the best performance and was saved as `mental_health_pipeline.pkl`.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web interface (`index.html`) |
| `GET` | `/docs` | Interactive Swagger API documentation |
| `POST` | `/predict` | Accepts student habit data and returns predicted score |

<details>
<summary><b>🔍 Click to view sample API Request & Response</b></summary>

#### Request (`POST /predict`):
```json
{
  "age": 21,
  "gender": "Male",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 3.5,
  "daily_unlocks": 45,
  "study_hours": 4.0,
  "physical_activity_hours": 1.5,
  "sleep_hours_per_night": 7.5,
  "stress_level": "Medium"
}
```

#### Response (`200 OK`):
```json
{
  "predicted_mental_health_score": 7.66
}
```

#### If invalid schedule is sent (`422 Unprocessable Entity`):
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body"],
      "msg": "Value error, Sleep (20.0h) + Physical Activity (1.0h) + Phone Usage (4.0h) = 25.0h, which exceeds 24 hours in a day. Please verify your daily schedule inputs."
    }
  ]
}
```
</details>

---

## 📁 Project Structure

```plaintext
Mental_Health_Score_Predictor/
├── ML_Project (1).ipynb                              # Data cleaning, EDA & model training notebook
├── Student Social Media And Mental Health Impact.csv # Dataset (5,000 records)
├── mental_health_pipeline.pkl                        # Serialized model & preprocessing pipeline
├── main.py                                           # FastAPI application & validation logic
├── index.html                                        # Web interface
├── requirements.txt                                  # Project dependencies
├── .gitignore                                        # Ignored files (caches, venvs)
└── README.md                                         # Project documentation
```
