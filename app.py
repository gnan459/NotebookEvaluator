import streamlit as st
import json
import nbformat
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from io import StringIO

# --- Helper Functions ---
def check_syntax(code):
    try:
        compile(code, "<string>", "exec")
        return True, None
    except SyntaxError as e:
        return False, str(e)

def evaluate_code_metrics_presence(code, problem_type):
    metrics_found = {}
    if problem_type == "titanic_classification":
        metrics_found["accuracy_found"] = "accuracy_score" in code
        metrics_found["f1_found"] = "f1_score" in code
    elif problem_type == "house_price_regression":
        metrics_found["mse_found"] = "mean_squared_error" in code
        metrics_found["mae_found"] = "mean_absolute_error" in code
        metrics_found["r2_found"] = "r2_score" in code
    return metrics_found

def evaluate_notebook_syntax(notebook_content):
    syntax_errors = 0
    for cell in notebook_content.cells:
        if cell.cell_type == "code":
            if not check_syntax(cell.source)[0]:
                syntax_errors += 1
    total_code_cells = sum(1 for cell in notebook_content.cells if cell.cell_type == "code")
    syntax_accuracy = (total_code_cells - syntax_errors) / (total_code_cells + 1e-6) * 100
    return syntax_accuracy

# --- Evaluation Function for CSV Comparison (Comparing All Rows) ---
def evaluate_predictions_csv(student_csv, reference_csv, problem_type):
    try:
        student_df = pd.read_csv(StringIO(student_csv))
        reference_df = pd.read_csv(StringIO(reference_csv))

        if problem_type == "titanic_classification":
            if 'PassengerId' not in student_df.columns or 'Survived' not in student_df.columns or \
               'PassengerId' not in reference_df.columns or 'Survived' not in reference_df.columns:
                return {"error": "Missing required columns in CSVs"}, 0

            merged_df = pd.merge(student_df[['PassengerId', 'Survived']], reference_df[['PassengerId', 'Survived']], on='PassengerId', suffixes=('_student', '_reference'))
            if len(merged_df) != len(reference_df):
                return {"error": "Number of rows in submission does not match reference"}, 0

            accuracy = accuracy_score(merged_df['Survived_reference'], merged_df['Survived_student'])
            f1 = f1_score(merged_df['Survived_reference'], merged_df['Survived_student'], average='binary')
            prediction_score = (accuracy + f1) / 2 * 100
            return {"accuracy": accuracy, "f1_score": f1}, prediction_score

        elif problem_type == "house_price_regression":
            if 'Id' not in student_df.columns or 'SalePrice' not in student_df.columns or \
               'Id' not in reference_df.columns or 'SalePrice' not in reference_df.columns:
                return {"error": "Missing required columns in CSVs"}, 0

            merged_df = pd.merge(student_df[['Id', 'SalePrice']], reference_df[['Id', 'SalePrice']], on='Id', suffixes=('_student', '_reference'))
            if len(merged_df) != len(reference_df):
                return {"error": "Number of rows in submission does not match reference"}, 0

            y_true = merged_df['SalePrice_reference']
            y_pred = merged_df['SalePrice_student']

            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            prediction_score = max(0, min(100, r2 * 100))

            return {"mse": mse, "mae": mae, "r2_score": r2}, prediction_score

        else:
            return {"error": "Invalid problem type for CSV evaluation"}, 0

    except Exception as e:
        return {"error": f"Error comparing CSVs: {e}"}, 0

# --- Main Streamlit App ---
st.title("Notebook Evaluator")

uploaded_notebook = st.file_uploader("Upload Student's .ipynb file", type="ipynb")
uploaded_submission_csv = st.file_uploader("Upload Student's Submission CSV (submission.csv)", type="csv")
problem_type = st.selectbox("Select the Problem Type:", ["titanic_classification", "house_price_regression"])
reference_csv_file = st.file_uploader(f"Upload Reference CSV for {problem_type.replace('_', ' ').title()}", type="csv")

if uploaded_notebook and uploaded_submission_csv and reference_csv_file and problem_type:
    try:
        notebook_content = nbformat.read(uploaded_notebook, as_version=4)
        syntax_accuracy = evaluate_notebook_syntax(notebook_content)
        metric_presence = evaluate_code_metrics_presence(
            "\n".join(cell.source for cell in notebook_content.cells if cell.cell_type == "code"),
            problem_type
        )
        predictions_results, predictions_score = evaluate_predictions_csv(
            uploaded_submission_csv.getvalue().decode('utf-8'),
            reference_csv_file.getvalue().decode('utf-8'),
            problem_type
        )
        # --- Calculate Final Score (Adjust Weights as Needed) ---
        syntax_weight = 0.3
        prediction_weight = 0.7
        final_score = (syntax_accuracy * syntax_weight) + (predictions_score * prediction_weight)

        evaluation_results = {
            "final_score": round(final_score, 2),
            "syntax_accuracy": round(syntax_accuracy, 2),
            "metric_presence": metric_presence,
            "prediction_evaluation": predictions_results,
            "prediction_score": round(predictions_score, 2),
        }

        st.subheader("Evaluation Results:")
        st.write(f"**Final Score:** {evaluation_results['final_score']:.2f}%")
        st.write(f"**Syntax Accuracy:** {evaluation_results['syntax_accuracy']:.2f}%")
        st.write("**Detected Metric Mentions:**")
        for metric, found in evaluation_results['metric_presence'].items():
            if found:
                st.write(f"- {metric.replace('_found', '').replace('_', ' ').title()}")
        st.write("**Prediction Evaluation:**")
        st.json(evaluation_results['prediction_evaluation'])
        st.write(f"**Prediction Score:** {evaluation_results['prediction_score']:.2f}%")

        json_output = json.dumps(evaluation_results, indent=4)
        st.download_button(
            label="Download Evaluation Results (JSON)",
            data=json_output,
            file_name="evaluation_results.json",
            mime="application/json",
        )

    except Exception as e:
        st.error(f"Error during evaluation: {e}")

elif uploaded_notebook or uploaded_submission_csv or reference_csv_file:
    st.warning("Please upload the student's notebook, submission CSV, and the reference CSV, and select the problem type to start evaluation.")