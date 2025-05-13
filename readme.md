
## 🧠 MedAI - Cancer Prediction Using AI
MedAI is an intelligent machine learning-based tool designed to predict cancer types using health data. It leverages multiple classification algorithms to analyze structured datasets (like CSVs) and identify the most accurate model for prediction.

## 🎯 Objective
MedAI automates the following:

✅ Reads medical/health data
✅ Trains and compares various ML models
✅ Visualizes and summarizes results
✅ Generates a professional PDF report

## 🤖 Machine Learning Models Used
The following models are trained and evaluated:

🔹 Logistic Regression

🌲 Random Forest

🧭 Support Vector Machine (SVM)

👥 K-Nearest Neighbors (KNN)

⚡ XGBoost

## 📊 What It Generates
After training, the tool produces:

📈 model_comparison.png: Bar graph comparing model accuracies

🧾 model_report.pdf: PDF summarizing results and confusion matrix

📉 confusion_matrix.png: Heatmap for the best-performing model

🧬 predicted_cancer_types.png: Count of predicted cancer types

## 🗂️ Project Structure
bash
Copy
Edit
MedAI/

├── data/
│   └── dataset.csv               # Input health dataset
├── main.py                       # Main script (training + reporting)
├── model_comparison.png          # Bar chart of model accuracies
├── confusion_matrix.png          # Confusion matrix (best model)
├── predicted_cancer_types.png    # Bar chart of cancer predictions
├── model_report.pdf              # Auto-generated PDF report
├── predictions_output.csv       # CSV of actual vs predicted labels
├── requirements.txt              # Required Python packages
└── README.md                     # Project documentation
## ⚙️ How to Use
## 📌 Step 1: Install Dependencies
Run this command to install all required libraries:

bash
Copy
Edit
pip install -r requirements.txt
📌 Step 2: Add Your Dataset
Place your CSV dataset in the data/ folder.

## 📝 Dataset Format Guidelines:

The target column (e.g., PULMONARY_DISEASE) should contain labels like "YES" or "NO".

The feature columns should include health indicators (e.g., age, symptoms, habits).

## 📌 Step 3: Run the Project
To start training and generating outputs, run:

bash
Copy
Edit
python main.py
## 📌 Step 4: Review Your Outputs
You will get the following:

model_comparison.png: Visual comparison of all model accuracies

confusion_matrix.png: For the best-performing model

predicted_cancer_types.png: Breakdown of predictions

model_report.pdf: Summary with accuracy + confusion matrix

predictions_output.csv: Full actual vs predicted labels (for the best model)

📎 [Sample Output Preview](https://docs.google.com/document/d/1FTj9zicWCrXGLfr9Ac-Q79WPvpC11W6uh_BPy-LVj7E/edit?usp=sharing)

## 📄 License
This project is licensed under the MIT License.






