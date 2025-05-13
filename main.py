import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load and preprocess dataset
def load_data():
    data = pd.read_csv('data/dataset.csv')  # Change path if needed
    X = data.drop(columns='PULMONARY_DISEASE')  # Features
    y = data['PULMONARY_DISEASE']  # Target

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # Convert 'NO'/'YES' to 0/1

    return X, y

# Plot model comparison
def plot_model_comparison(results):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title("Model Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()

# Plot predicted cancer types
def plot_predicted_cancer_types(y_pred):
    unique, counts = np.unique(y_pred, return_counts=True)
    plt.figure(figsize=(7, 5))
    plt.bar(unique, counts, color=['#FF6347', '#32CD32'])
    plt.title("Predicted Cancer Types")
    plt.xlabel("Cancer Type")
    plt.ylabel("Count")
    plt.xticks([0, 1], ['No Cancer', 'Cancer'])
    plt.tight_layout()
    plt.savefig("predicted_cancer_types.png")
    plt.show()

# Generate confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Cancer', 'Cancer'],
                yticklabels=['No Cancer', 'Cancer'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    plt.show()

# Generate PDF report
def generate_pdf_report(results, cm):
    c = canvas.Canvas("model_report.pdf", pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "MedAI - Full Dataset Model Report")
    c.drawString(100, 720, "Model Performance (Accuracy):")
    y_pos = 700
    for model, acc in results.items():
        c.drawString(100, y_pos, f"{model}: {acc:.4f}")
        y_pos -= 20

    c.drawString(100, y_pos - 20, "Confusion Matrix:")
    cm_str = f"[[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]"
    c.drawString(100, y_pos - 40, cm_str)
    c.save()

# Print accuracy bar chart in terminal
def print_accuracy_comparison(results):
    print("\nModel Accuracy Comparison")
    print("--------------------------")
    for model, accuracy in results.items():
        bar = '█' * int(accuracy * 10)
        print(f"{model:<20} | {bar:<20} {accuracy*100:.1f}%")

# Main function
def main():
    # Load data
    X, y = load_data()

    # Scale entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "XGBoost": xgb.XGBClassifier()
    }

    results = {}
    predictions = {}

    # Train and predict using full dataset
    for name, model in models.items():
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        acc = accuracy_score(y, y_pred)
        results[name] = acc
        predictions[name] = y_pred

    # Print model comparison
    print_accuracy_comparison(results)

    # Plot comparison bar graph
    plot_model_comparison(results)

    # Use best model for visualization
    best_model = max(results, key=results.get)
    best_y_pred = predictions[best_model]
    print(f"\nBest Model: {best_model}")

    # Plot predictions and confusion matrix
    plot_predicted_cancer_types(best_y_pred)
    plot_confusion_matrix(y, best_y_pred)

    # Generate PDF report
    generate_pdf_report(results, confusion_matrix(y, best_y_pred))

    # Print all predictions
    print("\nPredictions on Entire Dataset:")
    for i, (actual, pred) in enumerate(zip(y, best_y_pred)):
        print(f"Sample {i+1}: Actual = {actual}, Predicted = {pred}")

if __name__ == "__main__":
    main()
