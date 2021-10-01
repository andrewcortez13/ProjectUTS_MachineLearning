#line 2-6 (import library yang akan digunakan)
#numpy biasanya digunakan untuk pembentukan array data, matplotlib untuk membentuk visualisasi graph
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression

#data x,y
# input x biasanya berbentuk 2d dengan 2 column
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]] #data

#input y
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

#print data x, disini dapat dilihat bentuk datanya dari yang kita buat
print(x)
#print data y
print(y)

#membuat model regresi dari x,y 
#LinearRegression() pake dari library sklearn, membuat model linear regression dengan data x,y yang dimasukkan function fit
model = LinearRegression().fit(x,y) 

r_sq = model.score(x,y)
print('coefficient of determination : ', r_sq) #print coeffiencient of determintaion (R^2) 
print('intercept : ', model.intercept_) #print intercept, yang merupakan b0 dalam rumus
print('slope : ', model.coef_) #print slope,yang merupakan b1 dan b2 da;am rumus

#memprediksi data model
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

# x = np.concatenate((x,y.reshape(-1,1),y_pred.reshape(-1,1)), axis=1)
# df  =pd.DataFrame(x, columns=['x1','x2','y','y_predict'])
# print ("===========Data===========")
# print (df)

#membuat data baru (0-9),
x_new = np.arange(10).reshape((-1, 2))
print(x_new) #print data x
#var y_new memprediksi data dari x_new
y_new = model.predict(x_new)
print(y_new)

#coba graph
print ("Graph")
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x_new,y_new)
plt.show()