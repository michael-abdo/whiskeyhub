
# WhiskeyHub CLI-Based MVP: Data Sparsity Analysis + Linear Regression Model

## 📦 Prerequisites
Install required Python libraries:

```bash
pip install pandas sqlalchemy scikit-learn matplotlib seaborn
```

Update the DB connection string if using MySQL or PostgreSQL.

---

## ✅ STEP 1: Connect to SQL DB and Load Tables

### `db_connect.py`
```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine("sqlite:///whiskeyhub.db")  # Change for MySQL/Postgres
conn = engine.connect()

flights = pd.read_sql("SELECT * FROM flights", conn)
pours = pd.read_sql("SELECT * FROM flight_pours", conn)
notes = pd.read_sql("SELECT * FROM flight_notes", conn)
whiskeys = pd.read_sql("SELECT * FROM whishkeys", conn)

# Merge into one table
df = (pours
      .merge(notes, left_on="id", right_on="pour_id")
      .merge(flights, left_on="flight_id", right_on="id", suffixes=('', '_flight'))
      .merge(whiskeys, left_on="whiskey_id", right_on="id", suffixes=('', '_whiskey')))

df.to_csv("full_joined.csv", index=False)
print("✅ Merged data saved to full_joined.csv")
```

Run it:
```bash
python db_connect.py
```

---

## 📊 STEP 2: Check Data Sparsity

### `sparsity_analysis.py`
```python
import pandas as pd

df = pd.read_csv("full_joined.csv")
user_counts = df['user_id'].value_counts()
whiskey_counts = df['whiskey_id'].value_counts()

print(f"Users with < 3 tastings:\n{(user_counts < 3).sum()}")
print(f"Whiskeys with < 3 ratings:\n{(whiskey_counts < 3).sum()}")

print(f"\nTotal Ratings: {len(df)}")
print(f"Unique Users: {df.user_id.nunique()}")
print(f"Unique Whiskeys: {df.whiskey_id.nunique()}")

sparsity = 1 - (len(df) / (df.user_id.nunique() * df.whiskey_id.nunique()))
print(f"\n⚠️ Data Sparsity: {sparsity:.2%}")
```

Run it:
```bash
python sparsity_analysis.py
```

---

## 📈 STEP 3: Train a Simple Linear Regression Model

### `linear_model.py`
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("full_joined.csv")

# Aggregate ratings
df['rating'] = df.groupby(['user_id', 'whiskey_id'])['score'].transform('mean')
df = df.drop_duplicates(['user_id', 'whiskey_id'])

features = ['proof', 'price', 'age', 'viscosity', 'complexity']
df = df.dropna(subset=features + ['rating'])

X = df[features]
y = df['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("🔍 Evaluation Metrics:")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")
```

Run it:
```bash
python linear_model.py
```

---

## ✅ Final Output
- Sparsity % report
- User/whiskey coverage
- RMSE and R² model performance
- Pure CLI, SQL DB backend
