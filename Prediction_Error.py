"""
@预测误差计算
@Reversible data hiding based on reducing invalid shifting of pixels in histogram shifting
@基于减少直方图移动中像素无效移动的可逆数据隐藏
@安徽大学殷赵霞论文
"""
import tool
import numpy as np
import math
from matplotlib import pyplot as plt

#针对黑色区域进行像素值预测
def black_prediction_error(image):
    #计算图片的宽和高
    width,height = tool.get_w_h(image)
    #获取图片像素值
    pixels = tool.img_to_array(image)
    # 计算每个像素的预测误差（分为两部分进行计算， · 集以及 X 集合）
    predictValue = np.zeros((height,width))  # 创建一个大小与图片相同的二维数组，用来存放预测值 p''
    average = np.zeros((height,width))  # 创建一个大小与图片相同的二维数组，用来存放平均数 p'
    predictionError = np.zeros((height,width)) #创建一个大小与图片相同的二维数组，用来存放预测误差值 Pe
    predictionErrorlist = []  #用来存放所有的预测误差值
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 == 0:
                average[i][j] = math.floor((pixels[i - 1][j] + pixels[i + 1][j] + pixels[i][j - 1] + pixels[i][j + 1]) / 4)
                difference = np.zeros(4)
                difference[0] = abs(average[i][j] - pixels[i - 1][j])
                difference[1] = abs(average[i][j] - pixels[i + 1][j])
                difference[2] = abs(average[i][j] - pixels[i][j - 1])
                difference[3] = abs(average[i][j] - pixels[i][j + 1])
                weight1 = np.zeros(4)
                if sum(difference) == 0:
                    weight1[:] = 0.25
                else:
                    for x in range(4):
                        weight1[x] = sum(difference) / (1 + difference[x])
                weight = tool.normalization(weight1)
                predictValue[i][j] = math.floor(
                    weight[0] * pixels[i - 1][j] + weight[1] * pixels[i + 1][j] + weight[2] * pixels[i][j - 1] + weight[3] *
                    pixels[i][j + 1])
                predictionError[i][j] = pixels[i][j] - predictValue[i][j]
                predictionErrorlist.append(predictionError[i][j])
    #将预测误差转换为int型数组
    predictionErrorlist = np.array(predictionErrorlist,dtype=int)
    #计算预测误差值中每个元素出现的个数
    # cishu = tool.count(predictionErrorlist)
    # print(cishu)
    return predictionErrorlist,predictValue



def white_prediction_error(image):
    #计算图片的宽和高
    width,height = tool.get_w_h(image)
    #获取图片像素值
    pixels = tool.img_to_array(image)
    # 计算每个像素的预测误差（分为两部分进行计算， · 集以及 X 集合）
    predictValue = np.zeros((height,width))  # 创建一个大小与图片相同的二维数组，用来存放预测值 p''
    average = np.zeros((height,width))  # 创建一个大小与图片相同的二维数组，用来存放平均数 p'
    predictionError = np.zeros((height,width)) #创建一个大小与图片相同的二维数组，用来存放预测误差值 Pe
    predictionErrorlist = []  #用来存放所有的预测误差值
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 != 0:
                average[i][j] = math.floor((pixels[i - 1][j] + pixels[i + 1][j] + pixels[i][j - 1] + pixels[i][j + 1]) / 4)
                difference = np.zeros(4)
                difference[0] = abs(average[i][j] - pixels[i - 1][j])
                difference[1] = abs(average[i][j] - pixels[i + 1][j])
                difference[2] = abs(average[i][j] - pixels[i][j - 1])
                difference[3] = abs(average[i][j] - pixels[i][j + 1])
                weight1 = np.zeros(4)
                if sum(difference) == 0:
                    weight1[:] = 0.25
                else:
                    for x in range(4):
                        weight1[x] = sum(difference) / (1 + difference[x])
                weight = tool.normalization(weight1)
                predictValue[i][j] = math.floor(
                    weight[0] * pixels[i - 1][j] + weight[1] * pixels[i + 1][j] + weight[2] * pixels[i][j - 1] + weight[3] *
                    pixels[i][j + 1])
                predictionError[i][j] = pixels[i][j] - predictValue[i][j]
                predictionErrorlist.append(predictionError[i][j])
    #将预测误差转换为int型数组
    predictionErrorlist = np.array(predictionErrorlist,dtype=int)
    #计算预测误差值中每个元素出现的个数
    # print(tool.count(predictionErrorlist))

    return predictionErrorlist,predictValue
