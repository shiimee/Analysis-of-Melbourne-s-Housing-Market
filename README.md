# Analysis of Melbourne's Housing Market

This project asks one question about Melbourne's housing market: **what makes a property desirable?** It uses the [Kaggle Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) (~13,580 sales, 2016–2018). The notebook explores how distance from the CBD, property type, room count, amenities and sale method relate to price. It then builds a **suburb-relative desirability score (0–100)** that ranks each property against others in the same suburb. A linear regression tests how well the score predicts price. The score correlates with price at 0.53. As a predictor, it reaches R² ≈ 0.28 across all of Melbourne and roughly 0.68–0.74 within a single suburb. That makes it most useful for comparing a property against its local market, which helps buyers spot relative value and sellers price a listing.

## Key features

- **Data cleaning:** removes duplicate rows, drops redundant columns (`Bedroom2`, coordinates, `Postcode`, `CouncilArea`), turns zero land and building areas into NaN, and fixes data types.
- **Feature engineering:** adds 19 derived features, including `Total_amenities` (bathrooms + car spaces), `Is_house`, the interactions `Type_x_Distance` and `Type_x_Rooms`, distance bands, suburb median price, agent median price, and sale-date parts.
- **Exploratory analysis** answers four research questions:
  1. How distance from the CBD affects price, broken down by property type, region and suburb.
  2. How total amenities affect price.
  3. How room count and property type together affect price.
  4. How the sale method affects price: a one-hot-encoded regression with sold at auction (`S`) as the baseline.
- **Desirability score:** each property's percentile rank within its suburb for `Rooms` (35%) and `Total_amenities` (35%), plus a property-type weight (30%). The type weight is house = 1.0, townhouse = 0.3, unit = 0.0.
- **Evaluation:** a city-wide regression with an 80/20 train/test split, plus suburb-level regressions on randomly selected suburbs. Test scores use each suburb's training distribution, so test data doesn't leak into them. Results are reported as R², MAE, RMSE and residual plots.

## Tech stack

- Python 3 (developed on 3.13) and Jupyter Notebook
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn 1.4 or later, which provides `root_mean_squared_error`

## Setup

```bash
git clone https://github.com/shiimee/Analysis-of-Melbourne-s-Housing-Market.git
cd Analysis-of-Melbourne-s-Housing-Market

python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn "scikit-learn>=1.4" jupyter
```

The dataset is not in the repository. Download `melb_data.csv` from [Kaggle](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) and save it in the project root, next to the notebook.

## Running

```bash
jupyter notebook "Melb Housing Analysis (Kaggle).ipynb"
```

Run the cells from top to bottom. The suburb-level evaluation picks its training and test suburbs at random, so those numbers change from run to run.

## Example usage

Once the modelling cells have run, you can score properties and estimate prices directly in the notebook:

```python
# Add a Desirability_score column (0–100) to the engineered dataframe
scored = compute_desirability_score2(df_2)
scored.sort_values('Desirability_score', ascending=False)[
    ['Suburb', 'Type', 'Rooms', 'Total_amenities', 'Price', 'Desirability_score']
].head()

# Predicted price from the city-wide regression
predict_price(75.0)
# Desirability score 75.0 → predicted price: $1,318,346
```

## Limitations

- The data is limited to 2016–2018, and some suburbs have only 10–15 sales.
- The data has no school, public-transport or crime features. `BuildingArea` is missing for about 48% of rows, so the score leaves it out.
- The score's weights were set by hand, not learned from the data.
- Prediction error grows with price: the residual plots fan out at the high end.
