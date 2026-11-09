
import pandas as pd
import numpy as np
import joblib
import shap

from flask import Flask, request, jsonify, render_template
from db_config import get_db_connection

app = Flask(__name__)

# Load ML model and feature columns
model = joblib.load("land_acquisition_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Find the class representing delay
delay_class_index = list(model.classes_).index(1)


# Dashboard route
@app.route("/dashboard")
def dashboard():
    return render_template("index.html")


# Home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Land Acquisition Delay Prediction API is running!"
    })


# Convert a feature into a simple English explanation
def create_reason(feature, data):

    value = data.get(feature)

    if feature == "compensation_readiness_percentage":
        return (
            f"Compensation readiness is {value}%. "
            "Incomplete compensation preparation may delay payments "
            "to affected landowners."
        )

    elif feature == "land_records_verified_percentage":
        return (
            f"Land records verification is at {value}%. "
            "Unverified records may create documentation and "
            "ownership-related difficulties."
        )

    elif feature == "ownership_dispute_percentage":
        return (
            f"Ownership disputes account for {value}% of the project. "
            "Resolving disagreements over land ownership may take "
            "additional time."
        )

    elif feature == "litigation_cases":
        return (
            f"There are {value} pending litigation cases. "
            "Legal proceedings may slow down the land acquisition process."
        )

    elif feature == "public_objections_count":
        return (
            f"The project has {value} public objections. "
            "Addressing these objections may require additional time "
            "and discussions."
        )

    elif feature == "environmental_clearance_pending":
        if int(value) == 1:
            return (
                "Environmental clearance is still pending. "
                "The project may be delayed until the required approval "
                "is received."
            )
        return (
            "The model identified environmental clearance status "
            "as a contributing factor in this prediction."
        )

    elif feature == "forest_land_percentage":
        return (
            f"Forest land accounts for {value}% of the project area. "
            "Required forest-related approvals may take additional time."
        )

    elif feature == "utility_shifting_required":
        if int(value) == 1:
            return (
                "Utility shifting is required. Moving existing utilities "
                "may add extra work before land can be acquired or used."
            )
        return (
            "The model identified utility-shifting status as a factor "
            "in this prediction."
        )

    elif feature == "affected_households":
        return (
            f"The project affects {value} households. "
            "Coordinating with affected families may require additional time."
        )

    elif feature == "number_of_parcels":
        return (
            f"The project involves {value} land parcels. "
            "Verifying and processing multiple parcels may increase "
            "administrative work."
        )

    elif feature == "land_area_acres":
        return (
            f"The project covers {value} acres of land. "
            "The size of the land involved is one of the factors "
            "considered by the prediction model."
        )

    elif feature == "planned_duration_days":
        return (
            f"The planned project duration is {value} days. "
            "The model considers this timeline when estimating delay risk."
        )

    elif feature.startswith("project_type_"):
        project_type = feature.replace("project_type_", "")
        return (
            f"The project type is {project_type}. "
            "The model considers patterns associated with this project type."
        )

    elif feature.startswith("district_"):
        district = feature.replace("district_", "")
        return (
            f"The project is located in {district} district. "
            "The model considers district information in its prediction."
        )

    return (
        f"{feature.replace('_', ' ').title()} "
        "was identified as a factor influencing this prediction."
    )


# Generate a simple explanation using SHAP
def explain_prediction(project_data, original_data):

    shap_values = explainer.shap_values(project_data)

    # Extract SHAP values for the delay class
    if isinstance(shap_values, list):

        values = np.asarray(
            shap_values[delay_class_index]
        )[0]

    else:

        values = np.asarray(shap_values)

        if values.ndim == 3:
            values = values[0, :, delay_class_index]

        else:
            values = values[0]

    # Match features with their SHAP values
    contributions = []

    for feature, shap_value in zip(model_columns, values):

        shap_value = float(shap_value)

        if abs(shap_value) < 0.000001:
            continue

        contributions.append({
            "feature": feature,
            "shap_value": shap_value
        })

    # First select factors increasing delay risk
    increasing_factors = [
        item for item in contributions
        if item["shap_value"] > 0
    ]

    increasing_factors.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True
    )

    # Select the three strongest contributing factors
    selected = increasing_factors[:3]

    # If there are fewer than three positive factors,
    # include other important factors without misrepresenting them.
    if len(selected) < 3:

        remaining = [
            item for item in contributions
            if item not in selected
        ]

        remaining.sort(
            key=lambda item: abs(item["shap_value"]),
            reverse=True
        )

        selected.extend(
            remaining[:3 - len(selected)]
        )

    reasons = []

    for item in selected:

        feature = item["feature"]
        shap_value = item["shap_value"]

        sentence = create_reason(feature, original_data)

        if shap_value < 0:
            sentence = (
                "This factor reduces the predicted delay risk: "
                + sentence
            )

        reasons.append(sentence)

    if not reasons:
        return (
            "The model could not identify any significant individual "
            "factors for this prediction."
        )

    return " ".join(reasons)


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    connection = None
    cursor = None

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Please provide project details"
            }), 400

        # Required input fields
        required_fields = [
            "land_area_acres",
            "number_of_parcels",
            "affected_households",
            "ownership_dispute_percentage",
            "litigation_cases",
            "land_records_verified_percentage",
            "compensation_readiness_percentage",
            "environmental_clearance_pending",
            "forest_land_percentage",
            "public_objections_count",
            "utility_shifting_required",
            "planned_duration_days",
            "project_type",
            "district"
        ]

        missing = [
            field for field in required_fields
            if field not in data
        ]

        if missing:
            return jsonify({
                "error": "Missing fields",
                "fields": missing
            }), 400

        # Prepare project data
        new_project = pd.DataFrame([data])

        # Encode categorical features
        new_project = pd.get_dummies(new_project)

        # Match saved model columns
        new_project = new_project.reindex(
            columns=model_columns,
            fill_value=0
        )

        # Predict delay probability
        prediction_probabilities = model.predict_proba(
            new_project
        )[0]

        probability = prediction_probabilities[
            delay_class_index
        ]

        probability = round(float(probability) * 100, 2)

        # Determine risk category
        if probability >= 70:
            risk = "HIGH"

        elif probability >= 40:
            risk = "MEDIUM"

        else:
            risk = "LOW"

        # Generate simple explanation
        explanation = explain_prediction(
            new_project,
            data
        )

        # Connect to MySQL
        connection = get_db_connection()
        cursor = connection.cursor()

        # Save project details
        project_query = """
        INSERT INTO land_projects (
            project_name,
            land_area_acres,
            number_of_parcels,
            affected_households,
            ownership_dispute_percentage,
            litigation_cases,
            land_records_verified_percentage,
            compensation_readiness_percentage,
            environmental_clearance_pending,
            forest_land_percentage,
            public_objections_count,
            utility_shifting_required,
            planned_duration_days,
            project_type,
            district
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        project_values = (
            data.get("project_name", "Unnamed Project"),
            data["land_area_acres"],
            data["number_of_parcels"],
            data["affected_households"],
            data["ownership_dispute_percentage"],
            data["litigation_cases"],
            data["land_records_verified_percentage"],
            data["compensation_readiness_percentage"],
            data["environmental_clearance_pending"],
            data["forest_land_percentage"],
            data["public_objections_count"],
            data["utility_shifting_required"],
            data["planned_duration_days"],
            data["project_type"],
            data["district"]
        )

        cursor.execute(project_query, project_values)

        project_id = cursor.lastrowid

        # Save prediction history
        prediction_query = """
        INSERT INTO prediction_history (
            project_id,
            delay_probability,
            risk_category
        )
        VALUES (%s, %s, %s)
        """

        cursor.execute(
            prediction_query,
            (project_id, probability, risk)
        )

        connection.commit()

        # Return prediction and explanation
        return jsonify({
            "project_id": project_id,
            "delay_probability": probability,
            "risk_category": risk,
            "explanation": explanation,
            "message": "Prediction and explanation generated successfully!"
        })

    except Exception as e:

        if connection:
            connection.rollback()

        return jsonify({
            "error": str(e)
        }), 400

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
