PYTHON ?= python

.PHONY: demo pipeline test app install

install:
	$(PYTHON) -m pip install -r requirements.txt

demo: pipeline app

pipeline:
	$(PYTHON) run_all.py

test:
	$(PYTHON) -m pytest -q

app:
	$(PYTHON) -m streamlit run app.py
