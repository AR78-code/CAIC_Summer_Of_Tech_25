import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv('tweets_data.csv')

# This removes extreme outliers as visible in the graphs
Q1 = df['likes'].quantile(0.25)
Q3 = df['likes'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR
df = df[(df['likes'] >= lower_bound) & (df['likes'] <= upper_bound)]

X = df[['word_count', 'char_count', 'sentiment', 'has_media', 'company_encoded', 'hour', 'day_of_week']]
y = df['likes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds)
rmse = rmse ** 0.5
print("RMSE:", rmse)

import joblib
joblib.dump(model, 'like_predictor.pkl')
