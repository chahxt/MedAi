<!-- # PulmoPredict - Cancer Prediction Using AI

## Objective

PulmoPredict is a machine learning-based tool that helps predict cancer types from health data. The tool accepts structured datasets (e.g., CSV) containing various health indicators, trains multiple AI models, and evaluates their performance.

The tool compares models such as:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

It generates a bar graph comparing model performance, a confusion matrix, and a PDF report with the results.

## Project Structure

PulmoPredict/
├── data/
│ └── dataset.csv # Your input dataset
├── models/ # (Optional, if modularized)
├── main.py # Main script
├── utils.py # Helper functions (optional)
├── requirements.txt # Required libraries
├── README.md # Project description
├── model_comparison.png # Bar graph comparing model accuracy
├── confusion_matrix.png # Confusion matrix plot
└── model_report.pdf # Final PDF report


## How to Use

1. Install the required libraries using `pip install -r requirements.txt`.
2. Place your CSV dataset in the `data/` folder.
3. Run the project using the command `python main.py`.
4. The output will include:
   - Model comparison bar graph (`model_comparison.png`).
   - Confusion matrix (`confusion_matrix.png`).
   - PDF report (`model_report.pdf`).

## License

MIT License. -->


# MedAI - Cancer Prediction Using AI

## Objective

**MedAI** is a machine learning-based tool designed to predict cancer types from health data. It accepts structured datasets (e.g., CSV) containing various health indicators, trains multiple AI models, and evaluates their performance.

The tool compares the following machine learning models:
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **XGBoost**

It then generates:
- A bar graph comparing the accuracy of each model.
- A confusion matrix for the best-performing model.
- A PDF report summarizing the model performance and results.

## Project Structure

The project is organized as follows:

MedAI/
├── data/
│ └── dataset.csv # Your input dataset (CSV format)
├── models/ # (Optional, if you want to modularize models)
├── main.py # Main script for model training and evaluation
├── utils.py # Helper functions (optional, if needed)
├── requirements.txt # List of required libraries
├── README.md # Project description and usage instructions
├── model_comparison.png # Bar graph comparing model accuracy
├── confusion_matrix.png # Confusion matrix plot for the best model
├── predicted_cancer_types.png # Bar graph for predicted cancer types
└── model_report.pdf # Final PDF report with results and analysis

shell
Copy
Edit

## How to Use

### Step 1: Install Dependencies
Install the required libraries by running the following command:
<!-- ```bash
pip install -r requirements.txt -->
## Step 2: Prepare the Dataset
Place your CSV dataset in the data/ folder. The dataset should have the following structure:

The target variable (e.g., cancer type) should be labeled as Target or a similar name.

The feature columns should contain health-related indicators (age, smoking status, test results, etc.).

## Step 3: Run the Project
Execute the main script to train the models, evaluate them, and generate outputs:

bash
Copy
Edit
python main.py


## Step 4: Review the Outputs
After running the script, the following files will be generated:

model_comparison.png: A bar graph comparing the accuracy of each model.

confusion_matrix.png: A confusion matrix for the best-performing model.

predicted_cancer_types.png: A bar plot showing the predicted cancer types.

model_report.pdf: A PDF report summarizing the model results, including performance metrics and the confusion matrix.

## License
MIT License