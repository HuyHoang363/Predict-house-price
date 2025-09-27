import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

col_number = X.select_dtypes(include=['int', 'float']).columns.tolist()
print(f'Columns number: {col_number}')
col_string = X.select_dtypes(include=['object']).columns.to_list()
string_dict = {col: X[col].dropna().unique().tolist() for col in col_string}
print(f'String dict: {string_dict}')

# col_yesno = [col for col in col_string if set(X[col].str.lower().dropna().unique()) <= {'y', 'n'}]
# X[col_yesno] = X[col_yesno].apply(lambda col: col.map(lambda x: 1 if str(x).lower() == 'yes' else 0))

# Định nghĩa mapping cho các cột ordinal
ordinal_mappings = {
    'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtCond': ['Po', 'Fa', 'TA', 'Gd'],
    'BsmtExposure': ['No', 'Mn', 'Av', 'Gd'],
    'BsmtFinType1': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'BsmtFinType2': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'KitchenQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'FireplaceQu': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'PavedDrive': ['N', 'P', 'Y'],
    'Functional': ['Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
    'PoolQC': ['Fa', 'Gd', 'Ex'],
    'Fence': ['MnWw', 'MnPrv', 'GdWo', 'GdPrv']
}

# Lấy danh sách cột ordinal từ dict
ordinal_columns = list(ordinal_mappings.keys())
print("Ordinal columns:", ordinal_columns)

ordinal_categories = [ordinal_mappings[col] for col in ordinal_columns]
print(f'Ordinal categories: {ordinal_categories}')

nominal_columns = [col for col in col_string if col not in ordinal_columns]
print(f'Nominal: {nominal_columns}')

#Preprocess by columntransform
preprocess = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]),col_number),
    
        ('ord', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
        ]),ordinal_columns),
    
        ('nom', Pipeline([
            ('imputer', SimpleImputer(strategy = 'most_frequent')),
            ('nom', OneHotEncoder(handle_unknown='ignore'))
        ]), nominal_columns)
    ],
    remainder='drop'
)

#Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest = Pipeline([
    ('preprocess', preprocess),
    ('random_forest', RandomForestRegressor(random_state=42, n_jobs=-1))
])

#RandomSearchCV
param_dist = {
    'random_forest__n_estimators': [300, 400, 500, 600],
    'random_forest__max_depth': [3,4,5,6],
    'random_forest__min_samples_split': [3,5,7,9,11]
}

cv = KFold(n_splits=5, random_state=42, shuffle=True)

random_search = RandomizedSearchCV(
    random_forest,
    param_distributions=param_dist,
    cv=cv,
    n_jobs=-1,
    verbose=2,
    n_iter=10,
    random_state=42,
    scoring='neg_mean_squared_error'
)

random_search.fit(X_train, y_train)
print(f'Best param: {random_search.best_params_}')
print(f'Best score: {random_search.best_score_}')

best_model = random_search.best_estimator_
y_predict = best_model.predict(X_test)

print(f'MAE: {mean_absolute_error(y_test, y_predict)}')
print(f'MSE: {mean_squared_error(y_test, y_predict)}')
print(f'R2: {r2_score(y_test, y_predict)}')

#Test new data
new_data = pd.read_csv('test.csv')
# X_new = new_data[X.columns]
new_predict = best_model.predict(new_data)
new_data['PredictPrice'] = new_predict
new_data.to_csv('kaggle_predict.csv', index=False)

feature_names = best_model.named_steps['preprocess'].get_feature_names_out()
rf_model = best_model.named_steps['random_forest']
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values(by = 'importance', ascending=False)

print(importances.head(20))

# Vẽ biểu đồ
plt.figure(figsize=(10,6))
plt.barh(importances.head(20)['feature'], importances.head(20)['importance'])
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances - RandomForest")
plt.show()