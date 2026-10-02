.PHONY: all quick test app clean
all:
	python run_all.py
quick:
	python run_all.py --quick
test:
	pytest -q
app:
	streamlit run app/app.py
clean:
	rm -f data/*.csv models/*.joblib reports/*.json reports/figures/*.png
