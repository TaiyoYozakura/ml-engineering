import numpy as np

x = np.array([1, 2, 3, 4, 5])
#xII=np.array([60,65,70,80,90)])
y = np.array([12, 18, 25, 27, 35])

n=len(x)
sum_x  = np.sum(x)
sum_y  = np.sum(y)
sum_x2 = np.sum(x**2)
sum_xy = np.sum(x * y)

print(sum_x,sum_x2,sum_xy,sum_y)

xmean=np.mean(x)
ymean=np.mean(y)

m=(n*sum_xy-sum_x*sum_y)/(n*sum_x2-sum_x**2)
c=round(ymean-(m*xmean),2)

print(m,c)

y_dash=[]
for i in x:
    yd=(m*i)+c
    y_dash.append(yd)
print("Predicted Score :",np.array(y_dash))
def pred(i):
    return (m*i)+c
xI=int(input("Enter num of hours studied   "))
y=pred(xI)

