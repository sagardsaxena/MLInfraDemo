# Create a Virtual Environment
python3 -m venv venv
source venv/bin/activate

# Install Requirements
pip install --upgrade pip
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
pre-commit run --all-files
