# Run this file's commands one at a time in PowerShell.

Set-Location "c:\Users\ASUS ROG\OneDrive\ドキュメント\Agentic-Reasoning-System"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# Verify the solver.
python main.py --run-tests

# Start the Streamlit application.
streamlit run main.py

# Optional model/data workflow, after stopping Streamlit with Ctrl+C.
python scripts\eda.py
python src\prepare_dataset.py
python scripts\train_baseline.py
python src\train_model.py
python src\predict.py
python src\evaluate.py