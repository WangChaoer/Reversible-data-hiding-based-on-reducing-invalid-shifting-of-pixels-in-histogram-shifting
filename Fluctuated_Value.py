'''
@方差以及波动值计算
@实验部分
@主要内容：计算像素值的方差，并在方差的基础上计算除第一行和最后一行以及第一列和最后一列像素的波动值
'''
import numpy as np
import cv2
import tool


def blackcomplex(pixels):

    #计算图片的宽和高
    width,height = len(pixels[0]),len(pixels)
    #波动值数组的定义（不计算第一行和最后一行以及第一列和最后一列）
    com = np.zeros((height,width))
    #波动值的计算（不计算第一行和最后一行以及第一列和最后一列）
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 == 0:
                a = pixels[i-1][j]
                b = pixels[i][j-1]
                c = pixels[i][j+1]
                d = pixels[i+1][j]
                com[i][j] = abs(a-d)+abs(b-c)+abs(a+c-b-d)+abs(c+d-a-b)
    return com

def whitecomplex(pixels):
    #计算图片的宽和高
    width,height = len(pixels[0]),len(pixels)
    #波动值数组的定义（不计算第一行和最后一行以及第一列和最后一列）
    com = np.zeros((height,width))
    #波动值的计算（不计算第一行和最后一行以及第一列和最后一列）
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 != 0:
                a = pixels[i-1][j]
                b = pixels[i][j-1]
                c = pixels[i][j+1]
                d = pixels[i+1][j]
                com[i][j] = abs(a-d)+abs(b-c)+abs(a+c-b-d)+abs(c+d-a-b)
    return com

#计算黑色集（A集 ·集）每个像素的波动值（除去第一行第一列以及最后一行最后一列）
def black_calculate_fluctuated_value(pixels):

    #计算图片的宽和高
    width,height = len(pixels[0]),len(pixels)
    com = blackcomplex(pixels)
    #波动值数组的定义（不计算第一行和最后一行以及第一列和最后一列）
    flucatedValue = np.zeros((height,width))
    #波动值的计算（不计算第一行和最后一行以及第一列和最后一列）
    flucatedValueList = []
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 == 0:
                if i == 1 and j ==1:
                    flucatedValue[i][j] = com[i][j] + com[i + 1][j + 1]
                elif i ==1 and 1< j <width - 2:
                    flucatedValue[i][j] = com[i][j] + (com[i + 1][j + 1] + com[i + 1][j - 1]) / 2
                elif i ==1 and j == width - 2:
                    flucatedValue[i][j] = com[i][j] + com[i + 1][j - 1]
                elif 1<i<height - 2 and j==1 :
                    flucatedValue[i][j] = com[i][j] + (com[i - 1][j + 1] + com[i + 1][j + 1]) / 2
                elif 1<i<height - 2 and 1< j <width - 2 :
                    flucatedValue[i][j] = com[i][j] + (
                            com[i - 1][j - 1] + com[i - 1][j + 1] + com[i + 1][j - 1] + com[i + 1][j + 1]) / 4
                elif 1<i<height - 2 and j == width - 2:
                    flucatedValue[i][j] = com[i][j] + (com[i - 1][j - 1] + com[i + 1][j - 1]) / 2
                elif i==height - 2 and j == 1 :
                    flucatedValue[i][j] = com[i][j] + com[i - 1][j + 1]
                elif i == height - 2 and 1< j <width - 2 :
                    flucatedValue[i][j] = com[i][j] + (com[i - 1][j - 1] + com[i - 1][j + 1]) / 2
                elif i == height - 2 and j == width - 2:
                    flucatedValue[i][j] = com[i][j] + com[i - 1][j - 1]
                flucatedValueList.append(flucatedValue[i][j])
    flucatedValueList = np.array(flucatedValueList)

    return flucatedValueList

#计算白色集（B集 x集）每个像素的波动值（除去第一行第一列以及最后一行最后一列）
def white_calculate_fluctuated_value(pixels):

    #计算图片的宽和高
    width,height = len(pixels[0]),len(pixels)
    #获取图像每个像素点的方差
    com = whitecomplex(pixels)

    #波动值数组的定义（不计算第一行和最后一行以及第一列和最后一列）
    flucatedValue = np.zeros((height,width))
    #波动值的计算（不计算第一行和最后一行以及第一列和最后一列）
    flucatedValueList = []
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 != 0:
                if i == 1 and j ==1:
                    flucatedValue[i][j] = com[i][j] + com[i + 1][j + 1]
                elif i ==1 and 1< j <width - 2:
                    flucatedValue[i][j] = com[i][j] + (com[i + 1][j + 1] + com[i + 1][j - 1]) / 2
                elif i ==1 and j == width - 2:
                    flucatedValue[i][j] = com[i][j] + com[i + 1][j - 1]
                elif 1<i<height - 2 and j==1 :
                    flucatedValue[i][j] = com[i][j] + (com[i - 1][j + 1] + com[i + 1][j + 1]) / 2
                elif 1<i<height - 2 and 1< j <width - 2 :
                    flucatedValue[i][j] = com[i][j] + (
                            com[i - 1][j - 1] + com[i - 1][j + 1] + com[i + 1][j - 1] + com[i + 1][j + 1]) / 4
                elif 1<i<height - 2 and j == width - 2:
                    flucatedValue[i][j] = com[i][j] + (com[i - 1][j - 1] + com[i + 1][j - 1]) / 2
                elif i==height - 2 and j == 1 :
                    flucatedValue[i][j] = com[i][j] + com[i - 1][j + 1]
                elif i == height - 2 and 1< j <width - 2 :
                    flucatedValue[i][j] = com[i][j] + (com[i - 1][j - 1] + com[i - 1][j + 1]) / 2
                elif i == height - 2 and j == width - 2:
                    flucatedValue[i][j] = com[i][j] + com[i - 1][j - 1]
                flucatedValueList.append(flucatedValue[i][j])
    flucatedValueList = np.array(flucatedValueList)
    return flucatedValueList
