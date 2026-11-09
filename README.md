# Real-world Data Project: Retail Sales Analysis & Demand Prediction

End-to-end data science project (Retail domain): data cleaning, EDA, visualisations,
machine-learning prediction of daily units sold, and a written report with conclusions.

## How to run (3 steps)
```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows      |  source venv/bin/activate   (Mac/Linux)

# 2. install dependencies
pip install -r requirements.txt

# 3. run everything
python main.py
```
Runtime is about 1 minute. No internet connection or external dataset is needed:
a realistic retail dataset (21,930 rows, 5 stores x 6 categories x 2 years) is generated automatically
into `data/retail_sales.csv`.

## Use your own dataset (optional)
Provide a CSV with these columns and run `python main.py --data your_file.csv`:
`date, store, category, units_sold, unit_price, discount, promo, holiday, temperature, revenue`

## Project structure
```
main.py               entry point
requirements.txt
src/generate_data.py  synthetic retail data generator
src/preprocess.py     loading, validation, cleaning, date features
src/eda.py            7 EDA charts + key insights
src/model.py          feature engineering, 3 models + baseline, 3 charts
src/report.py         writes outputs/report.md
data/                 dataset (auto-created)
outputs/              report.md, figures/, model_results.csv, cleaned_data.csv, models/
```

## Results (sample run)
Gradient Boosting: R2 = 0.97, RMSE = 10.7 units, 57% better than the naive baseline.
Python 3.9+ recommended.
