import numpy as np
from group_lasso import  GroupLasso
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from group_lasso import LogisticGroupLasso

df=pd.read_csv('Question3Part1.csv',encoding='latin1')
df.columns = df.columns.str.strip()
print(df.columns)
test_df=df.iloc[100:]
train_df=df.iloc[:100]

## One-hot encode for categorical variables
train_transform=pd.get_dummies(train_df,columns=['motor','screw'],drop_first=False)
test_transform=pd.get_dummies(test_df,columns=['motor','screw'],drop_first=False)

test_transform=test_transform[train_transform.columns] ##ensure we have exactly same columns after one-hot encoding

X_train = train_transform.drop(columns='rise time').values #get only predictor variables
y_train = train_transform['rise time'].values #isolate response variable
X_test = test_transform.drop(columns='rise time').values
y_test = test_transform['rise time'].values

###Assigning a group to each feature (only group 0 and 1 have 5 features)
groups = np.array([2, 3] + [0]*5 + [1]*5)


lambda_grid=np.logspace(-4,-1,20)
optimal_lambda=None
best_mse=float('inf')
best_model=None

for lam in lambda_grid:
    GL = make_pipeline(
        StandardScaler(),
        GroupLasso(
            groups=groups,
            group_reg=lam,
            l1_reg=0.0,
            frobenius_lipschitz=True,
            scale_reg="group_size",
            supress_warning=True,
            n_iter=1000,
            tol=1e-3
        )
    )
    GL.fit(X_train, y_train)
    X_test_scaled = GL.named_steps['standardscaler'].transform(X_test)
    y_val_pred = GL.named_steps['grouplasso'].predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_val_pred)


    if mse <best_mse:
        best_mse=mse
        best_model=GL
        optimal_lambda=lam

scaler=GL.named_steps['standardscaler']
groplasso=GL.named_steps['grouplasso']
coefficients=groplasso.coef_

print("X_train shape:", X_train.shape)
print("Length of groups array:", len(groups))
print(train_transform.columns)



print(f"\n Optimal lambda (group_reg): {optimal_lambda:.4f}")
print(f"Test MSE: {best_mse:.4f}")
print("Non-zero Coefficients:\n", coefficients)

group_names = ['motor', 'screw', 'pgain', 'vgain']
print("\nGroup Importance (L2 norms):")
for g in range(4):
    group_indices = np.where(groups == g)[0]
    norm = np.linalg.norm(coefficients[group_indices])
    status = 'informative' if norm > 1e-6 else 'uninformative'
    print(f" - {group_names[g]} (Group {g}): L2 norm = {norm:.4f} → {status}")

###Part 2 - Logistic group lasso
df=pd.read_csv('Question3Part2.csv',encoding='latin1')
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
