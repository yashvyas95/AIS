import redpitaya_scpi as scpi
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.models import load_model
import tensorflow as tf
from FeatureExtraction import signal_feature
from BlackBox import LED_blinking

#Establishing SCPI connection
rp = scpi.scpi('192.168.128.1')
while 1:
    rp.tx_txt('ACQ:DEC 64') #DECIMATION FACTOR
    rp.tx_txt('ACQ:TRIG EXT_PE') #TRIGGER SOURCE
    rp.tx_txt('ACQ:TRIG:DLY 8192') #DELAY SET
    rp.tx_txt('ACQ:START') #ACQUISTION START
    while 1:
        rp.tx_txt('ACQ:TRIG:STAT?') #TRIGGER STATS
        if rp.rx_txt() == 'TD':
            break
    rp.tx_txt('ACQ:SOUR1:DATA?') #DATA FROM CHANNEL SOURCE1
    stri = rp.rx_txt()[1:-1]
    measRes = np.fromstring(stri,dtype=float,sep=',')
    plt.plot(measRes) #PLOT THE SIGNAL
    plt.show(block=False)
    time.sleep(5)
    plt.close("all")
    signal = np.array([measRes])
    f = open("Signal.csv", "a")
    np.savetxt(f,signal,fmt='%3.8f',delimiter=',') #SAVING THE SIGNAL
    f.close()
    model = tf.keras.models.load_model('lstm_model.h5')
    #model = load_model('lstm_model.h5')
    Data = np.array(measRes)
    Data1 = Data[4000:16384]
    signal_feature(Data) #FEATURE EXTRACTION
    Data1 = np.resize(Data1,(1,1,12384))
    #Data = np.resize(Data, (1, 16384, 1))
    prediction = model.predict_classes(Data1) #PREDICTION OF CLASS i.e WALL-0,HUMAN-1,CAR-2
    if prediction == 2:
        time.sleep(1/2.0)
        LED_blinking(2)
    elif prediction == 1:
        time.sleep(1/2.0)
        LED_blinking(1)
    elif prediction == 0:
        time.sleep(1/2.0)
        LED_blinking(0)
    
