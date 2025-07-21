import numpy as n 
import pandas as p 
import seaborn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plot
import torch 
import joblib 


df = p.read_csv(r"C:\Users\USER\Machinelearn\guest-regress=model_trainning\Housing.csv")

print(df.columns)

print(df.describe())

print(df["price"].unique())

print(df["price"].unique().sum())
# # kiểm tra giá trị vô hạn 
print(df.replace([n.inf, -n.inf], n.nan, inplace=True))
# # xử lý giá trị NaN
print(df.dropna(subset=['price'], inplace= True))
# # đặt kiểu thẩm mỹ 
seaborn.set_style("whitegrid")

# vẽ biểu đồ cần thiết 
plot.figure(figsize= (10, 6))
seaborn.scatterplot(x='area', y='price', data = df, hue='bedrooms', palette='viridis')
plot.title('biểu diễn giá nhà', fontsize = 12)
plot.xlabel('Area')
plot.ylabel('Price')
plot.legend(title = 'Bedrooms')
plot.show()

# lựa chọn cột cần 
df = df[['price'],['area'],['bedrooms'],['bathrooms']]

x = df[['area'],['bedrooms'],['bathrooms']]
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
# tính sai sót descent 
mse = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(mse,r2)
