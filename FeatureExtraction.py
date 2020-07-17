import numpy as np
from BlackBox import getAbsoluteEnergy,getFFt, getsumvalues

#data1 = np.array(genfromtxt('AllData/Wall.csv',delimiter=','))
#for i in range(data1.shape[0]):
#    shortData1 = np.array([data1[i][3500:]])
#    feature1 = getAbsoluteEnergy(shortData1[0])
#    feature2 = getsumvalues(shortData1[0])
#    feature3 = getSkewness(shortData1[0])
#    feature4 = getFFt(shortData1[0])
#    features = np.array([[feature1,feature2,feature3,feature4]])
#    f = open("AllData/WallFeatures.csv", "a")
#    np.savetxt(f,features,fmt='%3.8f',delimiter=',')
#    f.close()
#    
#data2 = np.array(genfromtxt('AllData/Human.csv',delimiter=','))
#for i in range(data2.shape[0]):
#    shortData2 = np.array([data2[i][3500:]])
#    feature1 = getAbsoluteEnergy(shortData2[0])
#    feature2 = getsumvalues(shortData2[0])
#    feature3 = getSkewness(shortData2[0])
#    feature4 = getFFt(shortData2[0])
#    features = np.array([[feature1,feature2,feature3,feature4]])
#    f = open("AllData/HumanFeatures.csv", "a")
#    np.savetxt(f,features,fmt='%3.8f',delimiter=',')
#    f.close()
#    
#data3 = np.array(genfromtxt('AllData/Car.csv',delimiter=','))
#for i in range(data3.shape[0]):
#    shortData3 = np.array([data3[i][3500:]])
#    feature1 = getAbsoluteEnergy(shortData3[0])
#    feature2 = getsumvalues(shortData3[0])
#    feature3 = getSkewness(shortData3[0])
#    feature4 = getFFt(shortData3[0])
#    features = np.array([[feature1,feature2,feature3,feature4]])
#    f = open("AllData/CarFeatures.csv", "a")
#    np.savetxt(f,features,fmt='%s',delimiter=',')
#    f.close()

def signal_feature(Data):
    shortD = Data[3500:]
    feature1 = getAbsoluteEnergy(shortD)
    feature2 = getsumvalues(shortD)
    feature3 = getFFt(shortD)
    features = np.array([[feature1,  feature2, feature3]])
    f = open("AllData/IncomingSignalFeatures.csv", "a")
    np.savetxt(f, features, fmt='%s', delimiter=',')
    f.close()
