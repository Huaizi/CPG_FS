import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn
import csv
import os
from dtw import dtw

x1=np.zeros(128)
x2=np.zeros(128)
x3=np.zeros(128)
x4=np.zeros(128)
x=np.zeros(128)
i=0
with open('data1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x1[i]=row[1]
        i=i+1

with open('data2.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x2[i]=row[1]
        i=i+1

with open('data3.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x3[i]=row[1]
        i=i+1

with open('data4.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x4[i]=row[1]
        i=i+1

euclidean_norm = lambda x, y: np.abs(x - y)
x1 = np.asarray(x1)
x2 = np.asarray(x2)
x3 = np.asarray(x3)
x4 = np.asarray(x4)

d, cost_matrix, acc_cost_matrix, path = dtw(x1, x2, dist=euclidean_norm)

print(d)

plt.imshow(acc_cost_matrix.T, origin='lower',  interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.show()


# plt.subplot(221)
# plt.plot(xf2,kf2,'r')
# plt.title('FFT of table1)',fontsize=10,color='#F08080')

# plt.subplot(222)
# plt.plot(xf2,yf2,'r')
# plt.title('FFT of table2)',fontsize=10,color='#F08080')

# plt.subplot(223)
# plt.plot(xf2,bary1)
# plt.title('simhash of table1)',fontsize=10,color='#F08080')

# plt.subplot(224)
# plt.plot(xf2,bary2,'b')
# plt.title('simhash of table2)',fontsize=10,color='#F08080')


# plt.show()

# # 采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
# # x=np.linspace(0,1,1400)      

# # 设置需要采样的信号，频率分量有180，390和600
# # y=7*np.sin(2*np.pi*180*x) + 2.8*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)
# y=x1
# yy=fft(y)                     #快速傅里叶变换
# yreal = yy.real               # 获取实数部分
# yimag = yy.imag               # 获取虚数部分

# k=x2
# kk=fft(k)
# kreal=kk.real
# kimg=kk.imag

# yf=abs(fft(y))                # 取绝对值
# yf1=abs(fft(y))/len(y)           #归一化处理
# yf2 = yf1[range(int(len(y)/2))]  #由于对称性，只取一半区间

# kf=abs(fft(k))
# kf1=abs(fft(k))/len(k)
# kf2=kf1[range(int(len(k)/2))]

# xf = np.arange(len(y))        # 频率
# xf1 = xf
# xf2 = xf[range(int(len(y)/2))]  #取一半区间

# cunchuy=yf2[0]
# bary1=np.zeros(12)
# bary2=np.zeros(12)
# for j in range(11):
#     if cunchuy<yf2[j+1]:
#         bary1[j]=1
#     else:
#         bary1[j]=0
#     cunchuy=yf2[j+1]
# bary1[11]=1

# cunchuk=kf2[0]
# for j in range(11):
#     if cunchuk<kf2[j+1]:
#         bary2[j]=1
#     else:
#         bary2[j]=0
#     cunchuk=kf2[j+1]
# bary2[11]=1

# sl=0
# bary3=np.zeros(12)
# for ij in range(11):
#     if bary1[ij]==bary2[ij]:
#         sl=sl+1
# print(sl/12)

