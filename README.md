# Predicting Healthcare Costs with Machine Learning

A data science project that explores drivers of healthcare costs and builds a model to predict medical charges using patient attributes.

## Tech Stack
- **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-learn**, **XGBoost**, **SHAP**
- **Streamlit** – interactive app  
- Jupyter Notebooks for data exploration and modeling

## Getting Started
Use the folder you already have on your machine.

1) (Optional but recommended) Create a virtual environment **in this folder** 
- python -m venv .venv 
- .venv\Scripts\activate    
2) Install dependencies:
```bash
pip install -r requirements.txt
```
3) Open the notebooks in `notebooks/` or launch the Streamlit app (see **Run the Project** below)

## Key Findings from EDA
- Smokers have significantly higher medical costs compared to non-smokers.
- Age has a moderate positive correlation with charges.
- High BMI increases charges mostly among smokers.
- Children and region don’t seem to have much impact.
- Medical charges are heavily skewed, with a few very high-cost outliers.

## Project Progress
- [x] Dataset loaded and cleaned
- [x] Exploratory data analysis (EDA)
- [x] Feature engineering
- [x] Model building & evaluation (linear regression, random forest, xgboost)
- [x] Streamlit app (live demo in the cloud)

## Notebooks
- [`01_eda.ipynb`](./notebooks/01_eda.ipynb) – Exploratory data analysis  
- [`02_preprocessing.ipynb`](./notebooks/02_preprocessing.ipynb) – Preprocessing work  
- [`03_modeling.ipynb`](./notebooks/03_modeling.ipynb) – Modeling work

## Model Performance Summary
We trained and compared three different models to predict medical charges:

| Model              | R² Score | MAE      | RMSE     |
|-------------------|----------|----------|----------|
| Linear Regression | 0.78     | $4,176   | $5,794   |
| Random Forest     | 0.86     | $2,719   | $4,704   |
| XGBoost           | 0.86     | **$2,665** | **$4,682** |

### Key Takeaways:
- **Linear Regression** gave a good starting point but couldn’t capture complex relationships.  
- **Random Forest** captured non-linear patterns better.  
- **XGBoost** slightly outperformed Random Forest and gave the most accurate predictions overall.

**Final Model Choice:** **XGBoost** was selected due to strong overall performance.

## Streamlit application
The app provides three tabs:
- **Predict Cost**: Input age, sex, BMI, region, smoker status, and children to get an estimated annual insurance charge.
- **What-If Analysis**: Create multiple profiles and compare predicted charges side-by-side with an interactive chart.
- **Explain Model**: Interactive SHAP dashboard (Plotly) with model selection and dependence plots.

## Credits
- Inspired by the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

## Run the Project

### Option A — Conda (recommended)
```bash
conda env create -f environment.yml
conda activate care-cost-predictor
streamlit run app/streamlit_app.py
```

### Option B — pip
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Project Structure
```
care-cost-predictor/
├─ data/                # Raw and cleaned datasets
├─ models/              # Trained model pickle files
├─ notebooks/           # EDA, preprocessing, and modeling notebooks
├─ app/                 # Streamlit application (+ theme in config.toml)
├─ docs/                # Screenshots & additional docs
├─ requirements.txt     # pip deps
├─ environment.yml      # conda env
├─ .gitignore           # ignore patterns
├─ .gitattributes       # Git LFS for large files
└─ README.md
```
