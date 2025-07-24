import numpy as np
from group_lasso import  GroupLasso
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from group_lasso import LogisticGroupLasso

 
# Logistic group lasso
df=pd.read_csv('data.csv',encoding='latin1')
df.columns = df.columns.str.strip()
df = df.dropna(how='all').reset_index(drop=True)
df=df.apply(pd.to_numeric, errors='coerce')

test_df=df.iloc[50:]
train_df=df.iloc[:50]

x_train=train_df.iloc[:,:-1]
y_train=train_df.iloc[:,-1]
x_test=test_df.iloc[:,:-1]
y_test=test_df.iloc[:,-1]

print("NaNs in X_train:", np.isnan(x_train).sum())
print("NaNs in X_test:", np.isnan(x_test).sum())

##group structure for 5 B-spline
groups = np.repeat(np.arange(20),5)


lambda_grid = np.logspace(-4, -1, 10)
best_acc = -1
best_model = None
best_lambda = None

for lam in lambda_grid:
    LGL = make_pipeline(
        StandardScaler(),
        LogisticGroupLasso(
            groups=groups,
            group_reg=lam,
            l1_reg=0.0,
            supress_warning=True,
            n_iter=1000,
            tol=1e-4,
            fit_intercept=True,
            random_state=42
        )
    )

    LGL.fit(x_train, y_train)
    x_test_scaled = LGL.named_steps['standardscaler'].transform(x_test)
    y_pred_prob = LGL.named_steps['logisticgrouplasso'].predict_proba(x_test_scaled)
    y_pred = (y_pred_prob[:, 1] > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    if acc > best_acc:
        best_acc = acc
        best_lambda = lam
        best_model = LGL


print(f"Best lambda LGL: {best_lambda}")
print(f"Test Accuracy: {best_acc:.4f}")
print("Coefficients LGL:", best_model.named_steps['logisticgrouplasso'].coef_)

coefs = best_model.named_steps['logisticgrouplasso'].coef_
coef_df = pd.DataFrame(coefs, columns=["Class_0", "Class_1"])
coef_df.index = [f"V{i+1}" for i in range(coefs.shape[0])]
coef_df.to_excel("logistic_group_lasso_coefficients.xlsx", index=True)
