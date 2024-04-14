cd documentation/pysa_tutorial

# use python3.8
python3.8 -m venv tutorial

source tutorial/bin/activate

# use pip instead of pip3
pip install pyre-check fb-sapp django-stubs 

# Before init make sure no .pysa_configuration file is there
pyre init-pysa

pyre analyze --no-verify --save-results-to ./pysa-runs

sapp analyze ./pysa-runs/taint-output.json

sapp server

