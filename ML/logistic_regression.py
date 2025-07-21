# bạn của linear regresion ( dự đoán giá trị liên tục ( thẳng băng ) ) là logistic regression (chữ Z) dùng để dự đoán giá nhà, thời tiếc, emai, thi đậu hay ko đậu
# hàm sigmoid f(x) = 1/ 
# quan trọng chỉ có chạy từ 0 đến 1
#mô hình logisitic bài tập#
import numpy as n  
from sklearn.linear_model import LogisticRegression 

# khởi tạo tham số
x = n.array([[20],[25],[30],[35],[40],[45],[50],[55],[60],[65]])
y = n.array([0,0,0,1,1,1,1,0,0,0])

# khởi tao mô hình 
model = LogisticRegression(solver='lbfgs')
model.fit(x,y)

#timhs toán dũw liệu
w_o = model.intercept_[0]
w_1 = model.coef_[0][0]
print(f"z = {w_o:.4f} +{w_1:.4f} * age")


age_to_predict = n.array([[35]])
prob = model.predict_proba(age_to_predict)[0][1]
prediction = 1 if prob >= 0.5 else 0

#test đáp án 
print(prediction)
print(prob)
