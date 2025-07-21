import matplotlib.pyplot as plot
from sklearn.metrics import mean_squared_error

# ---Giá xe ô tô 
car_age = [1, 2, 3, 4, 5, 6]
price = [20000, 18000, 15000, 12000, 9500, 8000]
pred_price = [21000, 17500, 15500, 11500, 9500, 8500]

mse_car = mean_squared_error(price, pred_price)
print("MSE (Car Price):", mse_car)

plot.figure(figsize=(10,6))
plot.scatter(car_age, price, label='Actual Price', color='blue')
plot.plot(car_age, pred_price, label='Prediction', color='red', marker='o')
plot.title('dự đoán giá xe')
plot.xlabel('Tuổi thọ của xe')
plot.ylabel('giá tiền USD')
plot.legend()
plot.grid(True)
plot.show()


# --- Điểm số học sinh 
study_hours = [2, 4, 6, 8, 10, 12, 14, 16]
actual_score = [50, 65, 70, 80, 85, 92, 95, 97]
predict_score =  [55, 60, 75, 78, 90, 88, 98, 100]

mse_score = mean_squared_error(actual_score, predict_score)
print("MSE (Student Score):", mse_score)

plot.figure(figsize=(10,6))
plot.scatter(study_hours, actual_score, label='Actual Score', color='blue')
plot.plot(study_hours, predict_score, label='Prediction', color='green', marker='o')
plot.title('dự đoán điểm học sinh')
plot.xlabel('Số giờ học ')
plot.ylabel('số điểm')
plot.legend()
plot.grid(True)
plot.show()


# --- Khách hàng và quảng cáo 
ad_spend = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
actual_cust = [50, 80, 120, 150, 200, 250, 300, 320, 350, 380]
predict_cust =   [55, 85, 110, 145, 195, 240, 290, 315, 340, 375]

mse_cust = mean_squared_error(actual_cust, predict_cust)
print("MSE (Customer Prediction):", mse_cust)

plot.figure(figsize=(10,6))
plot.scatter(ad_spend, actual_cust, label='Actual Customers', color='blue')
plot.plot(ad_spend, predict_cust, label='Prediction', color='purple', marker='o')
plot.title('dự đoán số chi tiêu của khách cho quảng cáo')
plot.xlabel('số chi tiêu cho quảng cáo')
plot.ylabel('số khách hàng')
plot.legend()
plot.grid(True)
plot.show()