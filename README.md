# Neural Network Admission Classifier

This project uses a neural network (MLPClassifier) to predict whether a student has a high chance of being admitted based on their GRE, TOEFL, CGPA, and other academic metrics.

## Project Structure

```
neural_network_classifier/
├── app.py                      # Streamlit web app
├── main.py                     # Pipeline runner
├── notebooks/
│   └── UCLA_Neural_Networks_Solution.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluate.py
│   └── utils.py
├── data/
│   └── Admission.csv           # <- Place the dataset here
├── requirements.txt
└── README.md
```

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add the dataset file `Admission.csv` to the `data/` folder.

3. Run the pipeline:
   ```bash
   python main.py
   ```

4. Launch the web app:
   ```bash
   streamlit run app.py
   ```

## Output

- Trains an MLP neural network
- Predicts admission chances
- Streamlit UI for live testing: https://ac-ml-nnc-hh-2025.streamlit.app/
