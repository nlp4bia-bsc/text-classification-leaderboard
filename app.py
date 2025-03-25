import os
import gradio as gr
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

testsets_root_path = "./testsets/"

# Function to load the dataset
def load_testsets(testsets_root_path: str) -> dict:
    datasets_dict = {}
    for ds in os.listdir(testsets_root_path):
        if ds.endswith(".csv"):  # Ensure only CSV files are processed
            csv_path = os.path.join(testsets_root_path, ds)
            df = pd.read_csv(csv_path)
            datasets_dict[ds.replace(".csv", "")] = df
    return datasets_dict

# Database setup
Base = declarative_base()

class Submission(Base):
    __tablename__ = 'submissions'
    id = Column(Integer, primary_key=True)
    dataset_name = Column(String)
    submission_name = Column(String)
    model_link = Column(String)
    person_name = Column(String)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1 = Column(Float)
    submission_date = Column(DateTime, default=datetime.utcnow)

engine = create_engine('sqlite:///submissions.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Function to fetch previous submissions for a selected dataset
def get_existing_submissions(dataset_name):
    existing_submissions = session.query(Submission).filter_by(dataset_name=dataset_name).order_by(
        Submission.submission_date.desc()).all()

    submissions_list = [{
        "Submission Name": sub.submission_name,
        "Model Link": sub.model_link,
        "Person Name": sub.person_name,
        "Accuracy": sub.accuracy,
        "Precision": sub.precision,
        "Recall": sub.recall,
        "F1": sub.f1,
        "Submission Date": sub.submission_date.strftime("%Y-%m-%d %H:%M:%S")
    } for sub in existing_submissions]

    return pd.DataFrame(submissions_list) if submissions_list else pd.DataFrame(columns=[
        "Submission Name", "Model Link", "Person Name", "Accuracy", "Precision", "Recall", "F1", "Submission Date"
    ])

# Evaluation function for text classification
def calculate_metrics(gs, pred):
    y_true = gs['label']
    y_pred = pred['label']
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        return accuracy, precision, recall, f1
    except:
        return None, None, None, None

def benchmark_interface(dataset_name, submission_file, submission_name, model_link, person_name):
    if not all([dataset_name, submission_file, submission_name, model_link, person_name]):
        return {"error": "All fields are required."}, pd.DataFrame()

    dataset_dict = load_testsets(testsets_root_path)
    df_gs = dataset_dict.get(dataset_name)
    if df_gs is None:
        return {"error": "Dataset not found."}, pd.DataFrame()

    # Parse the uploaded submission CSV
    submission_df = pd.read_csv(submission_file.name)

    # Ensure the columns are present
    if not all(col in submission_df.columns for col in ['file_name', 'label']):
        return {"error": "Submission file must contain 'file_name' and 'label' columns."}, pd.DataFrame()

    # Calculate metrics
    accuracy, precision, recall, f1 = calculate_metrics(gs=df_gs, pred=submission_df)
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
    if f1 is not None:
        # Save submission to the database
        new_submission = Submission(
            dataset_name=dataset_name,
            submission_name=submission_name,
            model_link=model_link,
            person_name=person_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1
        )
        session.add(new_submission)
        session.commit()

    # Fetch updated submissions
    submissions_df = get_existing_submissions(dataset_name)
    return metrics, submissions_df


def create_gradio_app():
    dataset_dict = load_testsets(testsets_root_path)
    dataset_names = list(dataset_dict.keys())

    with gr.Blocks() as demo:
        gr.Markdown("## Benchmarking Leaderboard for Text Classification")
        dataset_radio = gr.Radio(choices=dataset_names, label="Select Dataset")
        submission_file = gr.File(label="Upload Submission CSV")
        submission_name = gr.Textbox(label="Submission Name")
        model_link = gr.Textbox(label="Model Link on HuggingFace")
        person_name = gr.Textbox(label="Person Name")
        submit_button = gr.Button("Submit")
        metrics_output = gr.JSON(label="Evaluation Metrics")
        existing_submissions_output = gr.Dataframe(label="Existing Submissions")

        # When a dataset is selected, fetch previous submissions
        dataset_radio.change(
            fn=get_existing_submissions,
            inputs=[dataset_radio],
            outputs=[existing_submissions_output]
        )

        submit_button.click(
            fn=benchmark_interface,
            inputs=[dataset_radio, submission_file, submission_name, model_link, person_name],
            outputs=[metrics_output, existing_submissions_output]
        )
    return demo

def main():
    app = create_gradio_app()
    app.launch()

if __name__ == "__main__":
    main()
