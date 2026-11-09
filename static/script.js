
const form = document.getElementById("prediction-form");

const resultSection = document.getElementById("result");
const probabilityElement = document.getElementById("probability");
const riskElement = document.getElementById("risk");
const messageElement = document.getElementById("result-message");

const historyTable = document.getElementById("history-table");

const totalProjects = document.getElementById("total-projects");
const totalPredictions = document.getElementById("total-predictions");
const highRisk = document.getElementById("high-risk");
const mediumRisk = document.getElementById("medium-risk");

// Explainable AI section
const explanationContainer = document.getElementById("explanations");

let predictionCount = 0;
let highCount = 0;
let mediumCount = 0;
let lowCount = 0;

// Risk analytics chart
const chart = new Chart(
    document.getElementById("risk-chart"),
    {
        type: "doughnut",
        data: {
            labels: ["High Risk", "Medium Risk", "Low Risk"],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ["#e74c3c", "#f39c12", "#27ae60"],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: "bottom"
                }
            }
        }
    }
);

// Display SHAP explanations
function displayExplanations(explanations) {

    explanationContainer.innerHTML = "";

    if (!explanations || explanations.length === 0) {
        explanationContainer.textContent =
            "No explanations are available for this prediction.";
        return;
    }

    explanations.forEach(item => {

        const card = document.createElement("div");
        card.className = "explanation-card";

        const factor = document.createElement("h4");
        factor.textContent = item.factor;

        const impact = document.createElement("p");
        impact.textContent = item.impact;

        const value = document.createElement("p");
        value.textContent =
            "SHAP contribution: " + item.shap_value;

        card.appendChild(factor);
        card.appendChild(impact);
        card.appendChild(value);

        explanationContainer.appendChild(card);
    });
}

// Submit prediction form
form.addEventListener("submit", async function(event) {

    event.preventDefault();

    const button = document.getElementById("predict-btn");

    button.disabled = true;
    button.textContent = "Predicting...";

    const data = {
        project_name: document.getElementById("project_name").value,
        land_area_acres: Number(document.getElementById("land_area_acres").value),
        number_of_parcels: Number(document.getElementById("number_of_parcels").value),
        affected_households: Number(document.getElementById("affected_households").value),
        ownership_dispute_percentage: Number(document.getElementById("ownership_dispute_percentage").value),
        litigation_cases: Number(document.getElementById("litigation_cases").value),
        land_records_verified_percentage: Number(document.getElementById("land_records_verified_percentage").value),
        compensation_readiness_percentage: Number(document.getElementById("compensation_readiness_percentage").value),
        environmental_clearance_pending: Number(document.getElementById("environmental_clearance_pending").value),
        forest_land_percentage: Number(document.getElementById("forest_land_percentage").value),
        public_objections_count: Number(document.getElementById("public_objections_count").value),
        utility_shifting_required: Number(document.getElementById("utility_shifting_required").value),
        planned_duration_days: Number(document.getElementById("planned_duration_days").value),
        project_type: document.getElementById("project_type").value,
        district: document.getElementById("district").value
    };

    try {

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Prediction failed");
        }

        // Display prediction
        resultSection.hidden = false;

        probabilityElement.textContent = result.delay_probability + "%";
        riskElement.textContent = result.risk_category;
        messageElement.textContent = result.message;

        // Display Explainable AI results
        document.getElementById("explanations").textContent =
    result.explanation;

        // Update dashboard statistics
        predictionCount++;

        if (result.risk_category === "HIGH") {
            highCount++;
        } else if (result.risk_category === "MEDIUM") {
            mediumCount++;
        } else {
            lowCount++;
        }

        totalProjects.textContent = predictionCount;
        totalPredictions.textContent = predictionCount;
        highRisk.textContent = highCount;
        mediumRisk.textContent = mediumCount;

        // Update chart
        chart.data.datasets[0].data = [
            highCount,
            mediumCount,
            lowCount
        ];

        chart.update();

        // Update history table
        if (predictionCount === 1) {
            historyTable.innerHTML = "";
        }

        const row = document.createElement("tr");

        const values = [
            result.project_id,
            data.project_name,
            result.delay_probability + "%",
            result.risk_category,
            new Date().toLocaleString()
        ];

        values.forEach(value => {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.appendChild(cell);
        });

        historyTable.prepend(row);

        resultSection.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

    } catch (error) {

        alert("Error: " + error.message);

    } finally {

        button.disabled = false;
        button.textContent = "Predict Delay";
    }

});
