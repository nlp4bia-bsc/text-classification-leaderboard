# Text Classification Benchmark Leaderboard

This project provides a **leaderboard** for evaluating **Text Classification** models. Users can upload their model predictions in a CSV format, compare performance metrics against ground truth datasets, and track submissions over time.

## Features
- **Dataset Selection:** Users can choose a dataset from predefined test sets.
- **Submission Upload:** Supports CSV files with `file_name` and `label` columns.
- **Automated Evaluation:** Calculates **Accuracy, Precision, Recall, and F1-score**.
- **Leaderboard Tracking:** Stores and displays past experiments.
- **Gradio Interface:** Simple and interactive web interface.

## Requirements
Ensure you have the following installed before running the project:

```bash
pip install gradio pandas sqlalchemy scikit-learn
```

## Setup & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nlp4bia-bsc/text-classification-leaderboard.git
   cd text-classification-leaderboard
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Access the interface:**
   The application runs locally. Open your browser and go to:
   ```
   http://127.0.0.1:7860/
   ```

## Submission Format
Your submission file must be a **CSV** containing the following columns:

| file_name | label  |
|-----------|--------|
| doc1.txt  | spam   |
| doc2.txt  | ham    |
| doc3.txt  | spam   |

### Evaluation Metrics
The system calculates:
- **Accuracy**
- **Precision (weighted)**
- **Recall (weighted)**
- **F1-score (weighted)**

## Directory Structure
```
text-classification-leaderboard/
│── testsets/               # Folder containing test datasets
│── submissions.db          # SQLite database for storing results
│── app.py                  # Main application script
│── README.md               # Project documentation
```

## Future Improvements
- Add support for multi-label classification.
- Expand dataset compatibility with more formats.

## License
This project is licensed under the **MIT License**. Feel free to contribute and enhance it!

## Contributing
Pull requests are welcome! If you have suggestions or find issues, please open an issue on the repository.

---
**Author:** Wesam Alnabki
**GitHub:** [wesamalnabki](https://github.com/wesamalnabki)

