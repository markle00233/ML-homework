
import numpy as n

weight = 1
bias = 1

feature = n.array([5, 7, 9])
label = n.array([600, 790, 900])
predict = weight * feature + bias
print("predict: ", predict)
error = predict - label 
print("erorr: ", error)


#f(x) = x^2 + x0*sin(x), mô hình học thành công thì gọi là mô hình hội tụ 

#stochastic gradient descent, khởi tạo tham số -> suffle dữ liệu huấn -> tính gradient -> cập nhật tham số -> tối ưu

