# -*- coding: utf-8 -*-
__author__ = 'liudong'
__date__ = '2018/12/17 4:17 PM'
'''
李航统计学习方法第一章
我们用目标函数y=sin2PIx, 加上一个正态分布的噪音干扰，用多项式去拟合【例1.1 11页】
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq


# 目标函数
def real_func(x):
    return np.sin(2*np.pi*x)

# 多项式
def fit_func(p, x):
    f = np.poly1d(p)
    return x

# 残差
def residuals_func(p, x, y):
    ret = fit_func(p,x) - y
    return ret

# 十个点
x = np.linspace(0, 1, 10)
x_points = np.linspace(0,1,1000)
# 加上正态分布噪声的目标函数值
y_ = real_func(x)
y = [np.random.normal(0,0.1)+y1 for y1 in y_]

def fitting(M=0):
   """
   :param M:多项式的次数 
   :return: 
   """
   # 初始化多项式参数
   p_init = np.random.rand(M+1)
   # 最小二乘法
   p_lsq = leastsq(residuals_func, p_init, args=(x,y))
   print('Fitting Parameters:', p_lsq[0])

   plt.title("Figure of fit")
   plt.plot(x_points, real_func(x_points))
   plt.plot(x_points, fit_func(p_lsq[0], x_points))
   plt.plot(x, y, 'bo')
   # fitted curve 拟合曲线
   plt.legend(['real','fitted curve','noise'],loc = 'upper right')
   plt.show()
   return p_lsq

if __name__ == "__main__":
    p_lsq_3 = fitting (M=9)
