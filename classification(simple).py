from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
df=pd.read_csv(r"C:\Users\panch\Downloads\modified_real_estate_dataset.csv")
X=df.drop('Price',axis=1)
Y=df['Price']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
reg_model=RandomForestRegressor()
reg_model.fit(X_train,Y_train)
pred=reg_model.predict(X_test)
mse=mean_squared_error(Y_test,pred)
from sklearn.metrics import r2_score
print(r2_score(Y_test,pred))
print(f"Mean Squared Error: {mse}")
print("Model trained and evaluated successfully.")