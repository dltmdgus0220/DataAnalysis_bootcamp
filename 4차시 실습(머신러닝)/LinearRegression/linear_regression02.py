import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

rng = np.random.default_rng(42)
n = 200
x1 = rng.normal(size=n)
x2 = 0.9 * x1 + rng.normal(scale=0.1, size=n)
x3 = rng.normal(size=n)

df = pd.DataFrame({'x1':x1, 'x2':x2, 'x3':x3})
print(df.head())

def compute_vif(df_feature:pd.DataFrame) -> pd.DataFrame: # 데이터프레임을 받을 거고 리턴도 데이터프레임을 할거다(입력강제x). 내부적으로 성능도 좋아지고 가독성도 올라감.
    X = sm.add_constant(df_feature) # 절편 추가, 전부 1로 추가
    vif_rows = []
    for i in range(X.shape[1]):
        vif_val = variance_inflation_factor(X.values, i)
        vif_rows.append((X.columns[i], vif_val))
    return pd.DataFrame(vif_rows, columns=['features', 'VIF']).sort_values('VIF', ascending=False).reset_index(drop=True)

print()
vif_df = compute_vif(df)
print(vif_df) # 1-5:정상, 5-10:주의, 10이상:심각한 다중공선성
    
