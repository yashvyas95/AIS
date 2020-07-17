#!/usr/bin/python

import csv as csv
import numpy as np
from numpy import array
import pandas as pd
from scipy.fftpack import fft
import redpitaya_scpi as scpi
import time
# Load WallCsv File and Return Array[FloatArray]
#def LoadWallData():
#    with open('AllData/Wall.csv', 'r') as file:
#        reader = csv.reader(file)
#        row_Container = []
#        for row in reader:
#            row_Container.append((row))
#        return row_Container
#
## Load HumanCsv File and Return Array[FloatArray]
#def LoadHumanData():
#    with open('AllData/Human.csv', 'r') as file:
#        reader = csv.reader(file)
#        row_Container = []
#        for row in reader:
#            row_Container.append((row))
#        return row_Container
#
## Load CarCsv File and Return Array[FloatArray]
#def LoadCarData():
#    with open('AllData/Car.csv', 'r') as file:
#        reader = csv.reader(file)
#        row_Container = []
#        for row in reader:
#            row_Container.append((row))
#        return row_Container
#
## Get TotalData from CSV file and Return Array[FloatArray]
#def getData():
#    wallD = LoadWallData()
#    humanD = LoadHumanData()
#    carD = LoadCarData()
#    data = []
#    data.append(np.array(wallD))
#    data.append(np.array(humanD))
#    data.append(np.array(carD))
#    totalData = []
#    y = []
#    d = data[0]
#    h = data[1]
#    c = data[2]
#    for i in range(len(d)):
#        totalData.append(d[i])
#        y.append([0])
#    for i in range(len(h)):
#        totalData.append(h[i])
#        y.append([1])
#    for i in range(len(c)):
#        totalData.append(c[i])
#        y.append([2])
#    totalData = array(totalData)
#    totalData = np.resize(totalData, (len(totalData), 16834, 1))
#    totalLabel = array(y).T
#
#    return totalData,totalLabel
#
## Load WallFeature_abs File and Return Array[FloatArray]
#def LoadWallAbs():
#    with open('AllData/WallFeatures_abs.csv', 'r') as file:
#        reader = csv.reader(file)
#        rows = [];
#        for row in reader:
#            row = np.array(row)
#            rows.append(row.astype(np.float))
#        return rows
#
## Load CarFeature_abs File and Return Array[FloatArray]
#def LoadCarAbs():
#    with open('AllData/CarFeatures_abs.csv', 'r') as file:
#        reader = csv.reader(file)
#        rows = [];
#        for row in reader:
#            row = np.array(row)
#            rows.append(row.astype(np.float))
#        return rows
#
## Load HumanFeature_abs File and Return Array[FloatArray]
#def LoadHumanAbs():
#    with open('AllData/HumanFeatures_abs.csv', 'r') as file:
#        reader = csv.reader(file)
#        rows = [];
#        for row in reader:
#            row = np.array(row)
#            rows.append(row.astype(np.float))
#        return rows
#
## Load WallFeature_Skewness File and Return Array[FloatArray]
#def LoadWallSkewness():
#    with open('AllData/WallFeatures_skewness.csv', 'r') as file:
#        reader = csv.reader(file)
#        rows = [];
#        for row in reader:
#            row = np.array(row)
#            rows.append(row.astype(np.float))
#        return rows
#
## Load CarFeature_Skewness File and Return Array[FloatArray]
#def LoadCarSkewness():
#    with open('AllData/CarFeatures_skewness.csv', 'r') as file:
#        reader = csv.reader(file)
#        rows = [];
#        for row in reader:
#            row = np.array(row)
#            rows.append(row.astype(np.float))
#        return rows
#
## Load HumanFeature_Skewness File and Return Array[FloatArray]
#def LoadHumanSkewness():
#    with open('AllData/HumanFeatures_skewness.csv', 'r') as file:
#        reader = csv.reader(file)
#        rows = [];
#        for row in reader:
#            row = np.array(row)
#            rows.append(row.astype(np.float))
#        return rows
#
## Load WallFeature_SumoverTimeSeries File and Return Array[FloatArray]
#def LoadWallSum():
#    with open('AllData/WallFeatures_sum.csv', 'r') as file:
#        reader = csv.reader(file)
#        rows = [];
#        for row in reader:
#            row = np.array(row)
#            rows.append(row.astype(np.float))
#        return rows
#
## Load CarFeature_SumoverTimeSeries File and Return Array[FloatArray]
#def LoadCarSum():
#    with open('AllData/CarFeatures_sum.csv', 'r') as file:
#        reader = csv.reader(file)
#        rows = [];
#        for row in reader:
#            row = np.array(row)
#            rows.append(row.astype(np.float))
#        return rows
#
## Load HumanFeature_SumoverTimeSeries File and Return Array[FloatArray]
#def LoadHumanSum():
#    with open('AllData/HumanFeatures_sum.csv', 'r') as file:
#        reader = csv.reader(file)
#        rows = [];
#        for row in reader:
#            row = np.array(row)
#            rows.append(row.astype(np.float))
#        return rows
#
## Load WallFeature_FFT File and Return Array[FloatArray]
#def LoadWallFft():
#    with open('AllData/WallFeatures_fft.csv', 'r') as file:
#        reader = csv.reader(file)
#        rows = [];
#        for row in reader:
#            row = np.array(row)
#            rows.append(abs(row))
#        return rows
#
## Load CarFeature_FFT File and Return Array[FloatArray]
#def LoadCarFft():
#    with open('AllData/CarFeatures_sum.csv', 'r') as file:
#        reader = csv.reader(file)
#        rows = [];
#        for row in reader:
#            row = np.array(row)
#            rows.append(row.astype(np.float))
#        return rows
#
## Load HumanFeature_FFT File and Return Array[FloatArray]
#def LoadHumanFft():
#    with open('AllData/HumanFeatures_sum.csv', 'r') as file:
#        reader = csv.reader(file)
#        rows = [];
#        for row in reader:
#            row = np.array(row)
#            rows.append(row.astype(np.float))
#        return rows

#Get abosolute energy of TimeSeries
def getAbsoluteEnergy(TimeSeries):
    """
        E = \\sum_{i=1,\\ldots, n} x_i^2

    :param TimeSeries: the time series to calculate the feature of
    :type TimeSeries: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    # Checking Instance of TimeSeries
    x = np.asarray(TimeSeries)
    return np.dot(TimeSeries, TimeSeries)

# Calculate Skewness of Timeseries
def getSkewness(TimeSeries):
    """
    Returns the sample skewness of TimeSeries (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G1).

    :param TimeSeries: the time series to calculate the feature of
    :type TimeSeries: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    x = pd.Series(TimeSeries)
    return pd.Series.skew(TimeSeries)
#
##@set_property("fctype", "simple")
##@set_property("minimal", True)
def getsumvalues(x):
    """
    Calculates the sum over the time series values

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if len(x) == 0:
        return 0

    return np.sum(x)

#SCIPY FUNCTION FOR CALCULATION FFT OF TIMESERIES DATA
def getFFt(x):
    fft_x = fft(x)
    return fft_x

#Create Test and Train Data
#def creating_Test_Train_Data():
#    wall = LoadWallData()
#    human = LoadHumanData()
#    Car = LoadCarData()
#    lwall = len(wall)
#    lhuman = len(human)
#    lcar = len(Car)
#    test_x = []
#    train_x = []
#    test_y = []
#    train_y = []
#    for i in range(lwall):
#        if i < (148):
#            train_x.append(wall[i])
#            train_y.append(0)
#       #     f =open("TrainData.csv","a")
#       #     np.savetxt(f,np.array([wall[i]]),fmt='%s',delimiter=',')
#       #     f.close()
#       #     f1 = open("TrainDataLabel.csv","a")
#       #     np.savetxt(f1,np.array([0]),fmt='%s',delimiter=',')
#       #     f1.close()
#        else:
#            test_x.append(wall[i])
#            test_y.append(0)
#       #     f = open("TestData.csv", "a")
#       #     np.savetxt(f, np.array([wall[i]]), fmt='%s', delimiter=',')
#       #     f.close()
#       #     f1 = open("TestDataLabel.csv", "a")
#       #     np.savetxt(f1, np.array([0]), fmt='%s', delimiter=',')
#       #     f1.close()
#    for i in range(lhuman):
#        if i < (148):
#            train_x.append(human[i])
#            train_y.append(1)
#         #   f = open("TrainData.csv", "a")
#         #   np.savetxt(f, np.array([human[i]]), fmt='%s', delimiter=',')
#         #   f.close()
#         #   f1 = open("TrainDataLabel.csv", "a")
#         #   np.savetxt(f1, np.array([1]), fmt='%s', delimiter=',')
#         #   f1.close()
#        else:
#            test_x.append(human[i])
#            test_y.append(1)
#         #   f = open("TestData.csv", "a")
#         #   np.savetxt(f, np.array([human[i]]), fmt='%s', delimiter=',')
#         #   f.close()
#         #   f1 = open("TestDataLabel.csv", "a")
#         #   np.savetxt(f1, np.array([1]), fmt='%s', delimiter=',')
#         #   f1.close()
#    for i in range(len(Car)):
#        if i < (148):
#            train_x.append(Car[i])
#            train_y.append(2)
#      #      f = open("TrainData.csv", "a")
#       #     np.savetxt(f, np.array([Car[i]]), fmt='%s', delimiter=',')
#       #     f.close()
#       #     f1 = open("TrainDataLabel.csv", "a")
#      #      np.savetxt(f1, np.array([2]), fmt='%s', delimiter=',')
#     #       f1.close()
#        else:
#            test_x.append(Car[i])
#            test_y.append(2)
#    #        f = open("TestData.csv", "a")
#      #      np.savetxt(f, np.array([Car[i]]), fmt='%s', delimiter=',')
#     #       f.close()
#     #       f1 = open("TestDataLabel.csv", "a")
#     #       np.savetxt(f1, np.array([2]), fmt='%s', delimiter=',')
#    #        f1.close()
#    print(len(train_x),len(train_y))
#    print(len(test_x),len(test_y))
#
#    return train_x,train_y,test_x,test_y
#
#
def LED_blinking(number):
    rp_s = scpi.scpi('192.168.128.1')
    for i in range(5):
        time.sleep(1/2.0)
        rp_s.tx_txt('DIG:PIN LED' + str(number) + ',' + str(1))
        time.sleep(1/2.0)
        rp_s.tx_txt('DIG:PIN LED' + str(number) + ',' + str(0))
    
