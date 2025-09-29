import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('adult.csv', na_values=['?'])
df = df.drop(columns=['education'])
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

num_col = X.select_dtypes(include=['int', 'float']).columns.tolist()
print(f'Number columns: {num_col}')

string_col = X.select_dtypes(include=['object']).columns.tolist()
print(f'String columns: {string_col}')
string_list = {col: X[col].dropna().unique().tolist() for col in string_col}
print(f'String list: {string_list}')

preprocess = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_col),
        
        ('nom', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), string_col)
    ],
    remainder='drop'
)

#Split train test data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.35, random_state=42)

#Pipeline
random_forest = Pipeline([
    ('preprocess', preprocess),
    ('random_forest', RandomForestClassifier(random_state=42, n_jobs=-1))
])

#Cross validation
param_dist = {
    'random_forest__n_estimators': [300,400,500,600,700],
    'random_forest__max_depth': [3,4,5,6,7,9],
    'random_forest__min_samples_split': [5,6,7,8,9,10,11]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    random_forest,
    param_distributions=param_dist,
    cv=cv,
    n_iter = 10,
    n_jobs=-1,
    scoring = 'roc_auc',
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)
print(f'Best param: {random_search.best_params_}')
print(f'Best score: {random_search.best_score_}')

best_model = random_search.best_estimator_

y_predict = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print(f'Accuracy: {accuracy_score(y_test, y_predict)}')
print(f'Confusion matrix: {confusion_matrix(y_test, y_predict)}')
print(f'Classification report: {classification_report(y_test, y_predict)}')

#Draw heatmap
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predict')
plt.ylabel('Real')
plt.show()

#ROC-AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
AUC = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'AUC={AUC}')
plt.plot([0,1],[0,1],'--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.title('ROC-AUC for RF')
plt.show()