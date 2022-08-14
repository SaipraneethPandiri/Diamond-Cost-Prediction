#%%
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from math import sqrt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error 
import warnings
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math
warnings.filterwarnings("ignore")
df=pd.read_csv("C:\\Users\\jithi\\OneDrive\\Desktop\\vs_code\\ml_labs\\project\\diamonds.csv")
le = LabelEncoder()
df.cut = le.fit_transform(df.cut)
df.color = le.fit_transform(df.color)
df.clarity = le.fit_transform(df.clarity)
df=(df-df.mean())/df.std()
x = df.drop(['price'],axis=1)
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)
print(df.head())

#%%
#random forest algorithm
print('Random Forest Regressor Performance Metrics: ')
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
rf_pre =rf.predict(x_test)
rf_mse = mean_squared_error(y_test, rf_pre)
rf_mser=sqrt(rf_mse)
rf_maer = mean_absolute_error(y_test, rf_pre)
rf_r2r = r2_score(y_test, rf_pre)
rf_lse=((y_test-rf_pre)**2).sum()
print('R2            : ', rf_r2r)
print('mse            : ', rf_mse)
print('mser            : ', rf_mser)
print('lse            : ', rf_lse)

print("")
print("")

#%%

r2_k = []
print('K Nearest Neighbours  Performance Metrics: ')
for i in range(1,20):
    model=KNeighborsRegressor(n_neighbors=i)
    model.fit(x_train,y_train)
    knn_pre=model.predict(x_test)
    knn_mse = mean_squared_error(y_test, knn_pre)
    knn_mser=sqrt(knn_mse)
    knn_maer = mean_absolute_error(y_test, knn_pre)
    knn_r2r = r2_score(y_test, knn_pre)
    knn_lse=((y_test-knn_pre)**2).sum()
    r2_k.append(knn_r2r)
    print(f"Metrics for K = {i}: ")

    print('R2            : ', knn_r2r)
    print('mse            : ', knn_mse)
    print('mser            : ', knn_mser)
    print('lse            : ', knn_lse)

    print("")
    print("")
plt.plot(r2_k)
plt.show()
# %%
print('Neural Networks  Performance Metrics: ')
neural_model = MLPRegressor(hidden_layer_sizes=(10,9,8,6), activation='relu', solver='adam', max_iter=500)
neural_model.fit(x_train,y_train)
predict = neural_model.predict(x_test)
nn_mse = mean_squared_error(y_test, predict)
nn_mser=sqrt(nn_mse)
nn_maer = mean_absolute_error(y_test, predict)
nn_r2r = r2_score(y_test, predict)
nn_lse=((y_test-predict)**2).sum()

print('R2            : ', nn_r2r)
print('mse            : ', nn_mse)
print('mser            : ', nn_mser)
print('lse            : ', nn_lse)

print("")
print("")

# %%
data={'rf':rf_r2r,'knn':knn_r2r,'nn':nn_r2r}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
plt.bar(courses,values) 
plt.yscale("log")
plt.show()

 
# %%