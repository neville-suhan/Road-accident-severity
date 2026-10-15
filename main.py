
from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from xgboost import XGBClassifier


# --------------------------------------------------
# 1. PATHS AND APP CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PREPROCESSOR_PATH = BASE_DIR / "accident_preprocessor.joblib"
CLASSIFIER_PATH = BASE_DIR / "accident_classifier.json"
TEMPLATE_DIR = BASE_DIR / "templates"

app = FastAPI(title="RoadSafe AI - Accident Severity Prediction")

templates = Jinja2Templates(
    directory=str(TEMPLATE_DIR)
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Check that required model files exist.
for path in [PREPROCESSOR_PATH, CLASSIFIER_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required model file not found: {path}"
        )


# Load preprocessing pipeline.
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Load XGBoost classifier.
classifier = XGBClassifier()
classifier.load_model(str(CLASSIFIER_PATH))

logger.info(
    "Preprocessor and XGBoost classifier loaded successfully."
)


# --------------------------------------------------
# 2. MODEL FEATURES
# --------------------------------------------------

DEFAULT_FEATURES = [
    "Day_of_Week",
    "Junction_Control",
    "Junction_Detail",
    "Latitude",
    "Light_Conditions",
    "Local_Authority_(District)",
    "Longitude",
    "Number_of_Vehicles",
    "Police_Force",
    "Road_Surface_Conditions",
    "Road_Type",
    "Speed_limit",
    "Urban_or_Rural_Area",
    "Weather_Conditions",
    "Vehicle_Type",
    "Accident_Year",
    "Accident_Month",
    "Accident_Hour",
]

# Use the exact feature names and order from training.
if hasattr(preprocessor, "feature_names_in_"):
    FEATURES = list(preprocessor.feature_names_in_)
else:
    FEATURES = DEFAULT_FEATURES.copy()


# Identify numeric columns from the fitted preprocessor.
NUMERIC_COLUMNS = set()

if hasattr(preprocessor, "transformers_"):
    for name, transformer, columns in preprocessor.transformers_:
        if name == "num":
            NUMERIC_COLUMNS.update(list(columns))


# Fallback numeric columns.
if not NUMERIC_COLUMNS:
    NUMERIC_COLUMNS = {
        "Latitude",
        "Longitude",
        "Number_of_Vehicles",
        "Speed_limit",
        "Accident_Year",
        "Accident_Month",
        "Accident_Hour",
    }


OPTIONAL_FIELDS = {
    "Local_Authority_(District)",
    "Police_Force",
}


# --------------------------------------------------
# 3. CATEGORICAL VALUE MAPPINGS
# --------------------------------------------------

CATEGORY_CODES = {
    "Day_of_Week": {
        "sunday": 1,
        "monday": 2,
        "tuesday": 3,
        "wednesday": 4,
        "thursday": 5,
        "friday": 6,
        "saturday": 7,
    },

    "Road_Type": {
        "roundabout": 1,
        "one way street": 2,
        "one-way street": 2,
        "dual carriageway": 3,
        "single carriageway": 6,
        "slip road": 7,
        "unknown": 9,
    },

    "Junction_Control": {
        "none": 0,
        "give way or uncontrolled": 2,
        "stop sign": 3,
        "automatic traffic signal": 1,
        "auto traffic signal": 1,
        "authorised person": 4,
        "authorized person": 4,
        "data missing": -1,
        "data missing or out of range": -1,
    },

    "Junction_Detail": {
        "not at junction or within 20 metres": 0,
        "roundabout": 1,
        "mini-roundabout": 2,
        "mini roundabout": 2,
        "t or staggered junction": 3,
        "slip road": 5,
        "crossroads": 6,
        "more than 4 arms (not roundabout)": 7,
        "private drive or entrance": 8,
        "other junction": 9,
    },

    "Urban_or_Rural_Area": {
        "urban": 1,
        "rural": 2,
        "unallocated": 3,
    },

    "Road_Surface_Conditions": {
        "dry": 1,
        "wet or damp": 2,
        "snow": 3,
        "frost or ice": 4,
        "flood over 3cm deep": 5,
        "oil or diesel": 6,
        "mud": 7,
        "unknown": -1,
        "data missing": -1,
    },

    "Light_Conditions": {
        "daylight": 1,
        "darkness - lights lit": 4,
        "darkness - lights unlit": 5,
        "darkness - no lighting": 6,
        "darkness - lighting unknown": 7,
        "data missing": -1,
    },

    "Weather_Conditions": {
        "fine no high winds": 1,
        "fine without high winds": 1,
        "raining no high winds": 2,
        "raining without high winds": 2,
        "snowing no high winds": 3,
        "snowing without high winds": 3,
        "fine + high winds": 4,
        "fine with high winds": 4,
        "raining + high winds": 5,
        "raining with high winds": 5,
        "snowing + high winds": 6,
        "snowing with high winds": 6,
        "fog or mist": 7,
        "other": 8,
        "unknown": 9,
        "data missing": -1,
    },
}


def normalize_label(value):
    """Normalize a form label for matching."""
    return " ".join(
        str(value).strip().lower().split()
    )


def convert_value(column, value):
    """Convert form input to the expected input type."""

    if value is None or str(value).strip() == "":
        if column in OPTIONAL_FIELDS:
            return np.nan

        raise ValueError(
            f"Please provide a value for {column}."
        )

    value = str(value).strip()

    if column in NUMERIC_COLUMNS:
        try:
            return float(value)
        except ValueError:
            pass

        mapping = CATEGORY_CODES.get(column, {})
        normalized = normalize_label(value)

        if normalized in mapping:
            return float(mapping[normalized])

        raise ValueError(
            f"'{value}' is not a valid numeric value for "
            f"{column}. Please check the form options "
            "against the training data."
        )

    # Keep text categories as text if training used strings.
    return value


# --------------------------------------------------
# 4. PREPARE FORM INPUT
# --------------------------------------------------

def build_frame(form_data):
    """Build a DataFrame using training feature names."""

    row = {}

    for column in FEATURES:
        value = form_data.get(column)

        if value is None or str(value).strip() == "":
            if column in OPTIONAL_FIELDS:
                row[column] = np.nan
                continue

            raise ValueError(
                f"Missing required field: {column}"
            )

        row[column] = convert_value(column, value)

    return pd.DataFrame(
        [row],
        columns=FEATURES
    )


# --------------------------------------------------
# 5. PREDICTION AND PROBABILITIES
# --------------------------------------------------

SEVERITY_LABELS = {
    0: "Slight",
    1: "Serious",
    2: "Fatal",
}

ADVICE = {
    "Slight": (
        "The model predicts slight accident severity. "
        "Always follow road rules, maintain a safe distance, "
        "and drive according to road and weather conditions."
    ),

    "Serious": (
        "The model predicts serious accident severity. "
        "Reduce speed, remain alert, maintain a safe distance, "
        "and take extra care at junctions."
    ),

    "Fatal": (
        "The model predicts fatal accident severity. "
        "This is a model prediction, not a certainty. "
        "Drive cautiously, obey speed limits, and prioritize safety."
    ),
}


def predict_severity(input_df):
    """Preprocess input and predict accident severity."""

    # Step 1: Apply the fitted preprocessing pipeline.
    processed_input = preprocessor.transform(input_df)

    # Step 2: Predict using XGBoost.
    prediction = classifier.predict(processed_input)[0]

    if isinstance(prediction, np.generic):
        prediction = prediction.item()

    try:
        prediction = int(prediction)
    except (TypeError, ValueError):
        pass

    severity = SEVERITY_LABELS.get(
        prediction,
        str(prediction)
    )

    # Calculate class probabilities.
    probabilities = []

    if hasattr(classifier, "predict_proba"):
        probability_values = classifier.predict_proba(
            processed_input
        )[0]

        classes = getattr(
            classifier,
            "classes_",
            [0, 1, 2]
        )

        for class_value, probability in zip(
            classes,
            probability_values
        ):
            if isinstance(class_value, np.generic):
                class_value = class_value.item()

            label = SEVERITY_LABELS.get(
                class_value,
                str(class_value)
            )

            probabilities.append(
                (
                    label,
                    round(float(probability) * 100, 2)
                )
            )

        probabilities.sort(
            key=lambda item: item[1],
            reverse=True
        )

    return severity, probabilities


# --------------------------------------------------
# 6. HOME PAGE
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request
        }
    )


# --------------------------------------------------
# 7. PREDICTION FORM PAGE
# --------------------------------------------------

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "request": request,
            "error": None
        }
    )


# --------------------------------------------------
# 8. HANDLE FORM SUBMISSION
# --------------------------------------------------

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):

    form_data = dict(await request.form())

    try:
        # Prepare the form data.
        input_df = build_frame(form_data)

        # Generate prediction.
        severity, probabilities = predict_severity(input_df)

        # Prepare submitted-value summary.
        summary = [
            (key, form_data.get(key, "Not provided"))
            for key in FEATURES
            if form_data.get(key) not in (None, "")
        ]

        # FIX 1: Successful prediction response.
        return templates.TemplateResponse(
            request=request,
            name="result.html",
            context={
                "request": request,
                "severity": severity,
                "advice": ADVICE.get(
                    severity,
                    "Follow road safety rules and drive carefully."
                ),
                "probabilities": probabilities,
                "summary": summary,
            },
        )

    except (ValueError, TypeError) as exc:

        # FIX 2: Input validation error response.
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "request": request,
                "error": str(exc),
            },
            status_code=400,
        )

    except Exception:

        logger.exception(
            "Accident severity prediction failed"
        )

        # FIX 3: Unexpected error response.
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "request": request,
                "error": (
                    "Prediction failed. Please check the "
                    "model's feature types and the server "
                    "terminal for details."
                ),
            },
            status_code=500,
        )


# --------------------------------------------------
# 9. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "running",
        "preprocessor_loaded": preprocessor is not None,
        "classifier_loaded": classifier is not None,
    }
