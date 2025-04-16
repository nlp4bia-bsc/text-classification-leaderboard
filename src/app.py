import os
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Load environment variables ---
load_dotenv(find_dotenv())

# --- Set Page Config ---
st.set_page_config(page_title="Text Classification Leaderboard", layout="wide")

# --- Load Auth Config ---
config_path = os.getenv("USER_CONFIG_PATH", "DATA/users_config.yaml")
with open(config_path) as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- Login Widget ---
try:
    authenticator.login()
except Exception as e:
    st.error(e)

if st.session_state.get("authentication_status"):
    authenticator.logout("Logout", "sidebar")
    st.sidebar.write(f"Welcome *{st.session_state['name']}*")

    # --- Config ---
    testsets_root_path = os.getenv("TESTSETS_PATH", "DATA/testsets/")
    db_path = f"sqlite:///{os.getenv('DB_PATH', 'DATA/submissions_classification.db')}"
    submission_save_path = os.getenv("SUBMISSION_SAVE_PATH", "DATA/saved_submissions_classification/")
    os.makedirs(submission_save_path, exist_ok=True)

    # --- Database Setup ---
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

    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    def load_testsets(path):
        datasets = {}
        for ds in os.listdir(path):
            if ds.endswith(".csv"):
                df = pd.read_csv(os.path.join(path, ds))
                datasets[ds.replace(".csv", "")] = df
        return datasets

    def get_submissions(dataset):
        return session.query(Submission).filter_by(dataset_name=dataset).order_by(Submission.submission_date.desc()).all()

    def calculate_metrics(gs, pred):
        y_true = gs['label']
        y_pred = pred['label']
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            return accuracy, precision, recall, f1
        except Exception as e:
            st.error(f"Error in metric calculation: {e}")
            return None, None, None, None

    def display_leaderboard(dataset_name, dataset_dict):
        st.subheader("Leaderboard")
        submissions = get_submissions(dataset_name)
        if not submissions:
            st.info("No submissions yet.")
            return

        cols = st.columns([2, 2, 2, 1, 1, 1, 1, 2, 2])
        headers = ["Submission Name", "Model Link", "Person Name", "Accuracy", "Precision", "Recall", "F1", "Date", "Actions"]
        for col, header in zip(cols, headers): col.markdown(f"**{header}**")

        for sub in submissions:
            cols = st.columns([2, 2, 2, 1, 1, 1, 1, 2, 2])
            cols[0].markdown(f"**{sub.submission_name}**")
            cols[1].markdown(sub.model_link)
            cols[2].markdown(sub.person_name)
            cols[3].markdown(f"{sub.accuracy:.2f}")
            cols[4].markdown(f"{sub.precision:.2f}")
            cols[5].markdown(f"{sub.recall:.2f}")
            cols[6].markdown(f"{sub.f1:.2f}")
            cols[7].markdown(sub.submission_date.strftime("%Y-%m-%d %H:%M:%S"))

            delete_col, reeval_col, download_col = cols[8].columns([1, 1, 1])

            if delete_col.button("🗑️", key=f"del_{sub.id}"):
                # Delete submission from DB
                session.delete(sub)
                session.commit()

                # Delete corresponding CSV file
                csv_file_path = os.path.join(submission_save_path, f"{sub.dataset_name}__{sub.submission_name}.csv")
                try:
                    if os.path.exists(csv_file_path):
                        os.remove(csv_file_path)
                except Exception as e:
                    st.warning(f"CSV file could not be deleted: {e}")

                st.success(f"Deleted: {sub.submission_name}")
                st.rerun()

            if reeval_col.button("🔁", key=f"reeval_{sub.id}"):
                path = os.path.join(submission_save_path, f"{sub.dataset_name}__{sub.submission_name}.csv")
                if not os.path.exists(path):
                    st.error("File not found.")
                else:
                    gs = dataset_dict.get(sub.dataset_name)
                    pred = pd.read_csv(path)
                    acc, prec, rec, f1 = calculate_metrics(gs, pred)
                    st.subheader("Re-evaluation Results")
                    st.json({
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1": f1
                    })
                    st.success("Re-evaluated.")

            file_path = os.path.join(submission_save_path, f"{sub.dataset_name}__{sub.submission_name}.csv")
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    download_col.download_button("⬇️", f, file_name=os.path.basename(file_path))

    def submit_section(dataset_name, dataset_dict):
        st.subheader("Submit Your Model Prediction")
        with st.form("submission_form"):
            file = st.file_uploader("Upload CSV", type=["csv"])
            name = st.text_input("Submission Name")
            link = st.text_input("Model Link")
            person = st.text_input("Your Name")
            submit = st.form_submit_button("Submit")

            if submit:
                if not all([file, name, link, person]):
                    st.error("All fields required.")
                    return
                try:
                    gs_df = dataset_dict.get(dataset_name)
                    save_name = f"{dataset_name}__{name}.csv"
                    save_path = os.path.join(submission_save_path, save_name)
                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())
                    file.seek(0)
                    pred_df = pd.read_csv(file)
                    if not all(col in pred_df.columns for col in ['file_name', 'label']):
                        st.error("CSV must have 'file_name' and 'label' columns.")
                        return

                    acc, prec, rec, f1 = calculate_metrics(gs_df, pred_df)

                    session.query(Submission).filter_by(dataset_name=dataset_name, submission_name=name).delete()
                    new_sub = Submission(
                        dataset_name=dataset_name,
                        submission_name=name,
                        model_link=link,
                        person_name=person,
                        accuracy=acc, precision=prec, recall=rec, f1=f1
                    )
                    session.add(new_sub)
                    session.commit()
                    st.session_state['last_metrics'] = {
                        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1
                    }
                    st.success("Submitted and evaluated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Submission failed: {e}")

    def main():
        dataset_dict = load_testsets(testsets_root_path)
        if not dataset_dict:
            st.warning("No testsets found.")
            return

        dataset_name = st.selectbox("Choose Dataset", list(dataset_dict.keys()))
        display_leaderboard(dataset_name, dataset_dict)
        submit_section(dataset_name, dataset_dict)

        if 'last_metrics' in st.session_state:
            st.subheader("Last Evaluation")
            st.json(st.session_state.pop('last_metrics'))

    if __name__ == "__main__":
        main()

elif st.session_state.get("authentication_status") is False:
    st.error("Username/password is incorrect")
elif st.session_state.get("authentication_status") is None:
    st.warning("Please enter your username and password")
