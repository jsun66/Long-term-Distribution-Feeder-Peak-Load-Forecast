garort # Importing the libraries
import numpy as np
from numpy import array
from numpy import reshape
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.optimizers import Adam
from keras.layers import Dropout
import time

data=pd.read_excel('Loading Summary.xlsx',sheet_name="Summer",usecols = "B:P")

ResDat2016=pd.read_excel('2009-2016 Site Count.xlsx',sheet_name="2016",usecols = "B,I")
ResDat2015=pd.read_excel('2009-2016 Site Count.xlsx',sheet_name="2015",usecols = "B,I")
ResDat2014=pd.read_excel('2009-2016 Site Count.xlsx',sheet_name="2014",usecols = "B,I")
ResDat2013=pd.read_excel('2009-2016 Site Count.xlsx',sheet_name="2013",usecols = "B,I")
ResDat2012=pd.read_excel('2009-2016 Site Count.xlsx',sheet_name="2012",usecols = "B,I")
ResDat2011=pd.read_excel('2009-2016 Site Count.xlsx',sheet_name="2011",usecols = "B,I")
ResDat2010=pd.read_excel('2009-2016 Site Count.xlsx',sheet_name="2010",usecols = "B,I")
ResDat2009=pd.read_excel('2009-2016 Site Count.xlsx',sheet_name="2009",usecols = "B,I")
ResList2016 = ResDat2016['Rate'].tolist()
ResList2015 = ResDat2015['Rate'].tolist()
ResList2014 = ResDat2014['Rate'].tolist()
ResList2013 = ResDat2013['Rate'].tolist()
ResList2012 = ResDat2012['Rate'].tolist()
ResList2011 = ResDat2011['Rate'].tolist()
ResList2010 = ResDat2010['Rate'].tolist()
ResList2009 = ResDat2009['Rate'].tolist()
print ('ResData Complete')

LT2018=pd.read_excel('Load Transfer Summary S.xlsx',sheet_name="2018",usecols = "B,G")
LT2017=pd.read_excel('Load Transfer Summary S.xlsx',sheet_name="2017",usecols = "B,G")
LT2016=pd.read_excel('Load Transfer Summary S.xlsx',sheet_name="2016",usecols = "B,G")
LT2015=pd.read_excel('Load Transfer Summary S.xlsx',sheet_name="2015",usecols = "B,G")
LT2014=pd.read_excel('Load Transfer Summary S.xlsx',sheet_name="2014",usecols = "B,G")
LT2013=pd.read_excel('Load Transfer Summary S.xlsx',sheet_name="2013",usecols = "B,G")
LT2012=pd.read_excel('Load Transfer Summary S.xlsx',sheet_name="2012",usecols = "B,G")
LTList2018=LT2018['Feeder'].tolist()
LTList2017=LT2017['Feeder'].tolist()
LTList2016=LT2016['Feeder'].tolist()
LTList2015=LT2015['Feeder'].tolist()
LTList2014=LT2014['Feeder'].tolist()
LTList2013=LT2013['Feeder'].tolist()
LTList2012=LT2012['Feeder'].tolist()
print ('LT Complete')

Add2018=pd.read_excel('Large Load Addition Summary S.xlsx',sheet_name="2018",usecols = "E,N")
Add2017=pd.read_excel('Large Load Addition Summary S.xlsx',sheet_name="2017",usecols = "E,N")
Add2016=pd.read_excel('Large Load Addition Summary S.xlsx',sheet_name="2016",usecols = "E,N")
Add2015=pd.read_excel('Large Load Addition Summary S.xlsx',sheet_name="2015",usecols = "D,M")
Add2014=pd.read_excel('Large Load Addition Summary S.xlsx',sheet_name="2014",usecols = "D,M")
Add2013=pd.read_excel('Large Load Addition Summary S.xlsx',sheet_name="2013",usecols = "D,M")
Add2012=pd.read_excel('Large Load Addition Summary S.xlsx',sheet_name="2012",usecols = "D,M")
Add2011=pd.read_excel('Large Load Addition Summary S.xlsx',sheet_name="2011",usecols = "C,F")
Add2010=pd.read_excel('Large Load Addition Summary S.xlsx',sheet_name="2010",usecols = "A,C")
Add2009=pd.read_excel('Large Load Addition Summary S.xlsx',sheet_name="2009",usecols = "A,C")
AddList2018=Add2018['Feeder'].tolist()
AddList2017=Add2017['Feeder'].tolist()
AddList2016=Add2016['Feeder'].tolist()
AddList2015=Add2015['Feeder'].tolist()
AddList2014=Add2014['Feeder'].tolist()
AddList2013=Add2013['Feeder'].tolist()
AddList2012=Add2012['Feeder'].tolist()
AddList2011=Add2011['Feeder'].tolist()
AddList2010=Add2010['Feeder'].tolist()
AddList2009=Add2009['Feeder'].tolist()
print ('LT Complete')

X=pd.DataFrame(columns=['Load','GDP','MaxT','T','ResRatio','LT','Add'])
Y=pd.DataFrame(columns=['NLoad'])

for i in range (0,227):
    print (i)
    feedername=data.Feeder[i]
    if feedername[0]=="8": 
        if len(feedername)==6:
            feedername1="008-0"+feedername[2:6]
        else:
            feedername1="00"+feedername
    if feedername[0]=="2": 
        if len(feedername)==8:
            feedername1="025-0"+feedername[3:8]
        else:
            feedername1="0"+feedername
    ResRatio2016=0.63 #default value
    ResRatio2015=0.63
    ResRatio2014=0.63
    ResRatio2013=0.63
    ResRatio2012=0.63
    ResRatio2011=0.63
    ResRatio2010=0.63        
    ResRatio2009=0.63 #default value
    LT17=0 #default value
    LT16=0
    LT15=0
    LT14=0
    LT13=0
    LT12=0
    Add17=0        
    Add16=0 #default value
    Add15=0 #default value
    Add14=0
    Add13=0
    Add12=0
    Add11=0
    Add10=0
    Add09=0        

    if feedername1 in ResList2016:
        ResRatio2016=ResDat2016[ResDat2016.Rate==feedername1].iloc[0,1] #Use "==" and iloc to obtain the residential ratio
    if feedername1 in ResList2015:
        ResRatio2015=ResDat2015[ResDat2015.Rate==feedername1].iloc[0,1]
    if feedername1 in ResList2014:
        ResRatio2014=ResDat2014[ResDat2014.Rate==feedername1].iloc[0,1]
    if feedername1 in ResList2013:
        ResRatio2013=ResDat2013[ResDat2013.Rate==feedername1].iloc[0,1]
    if feedername1 in ResList2012:
        ResRatio2012=ResDat2012[ResDat2012.Rate==feedername1].iloc[0,1]
    if feedername1 in ResList2011:
        ResRatio2011=ResDat2011[ResDat2011.Rate==feedername1].iloc[0,1] #Use "==" and iloc to obtain the residential ratio
    if feedername1 in ResList2010:
        ResRatio2010=ResDat2010[ResDat2010.Rate==feedername1].iloc[0,1]
    if feedername1 in ResList2009:
        ResRatio2009=ResDat2009[ResDat2009.Rate==feedername1].iloc[0,1]

    if feedername in LTList2017:
        LT17=sum(LT2017[LT2017.Feeder==feedername].iloc[:,1])
    if feedername in LTList2016:
        LT16=sum(LT2016[LT2016.Feeder==feedername].iloc[:,1])
    if feedername in LTList2015:
        LT15=sum(LT2015[LT2015.Feeder==feedername].iloc[:,1])
    if feedername in LTList2014:
        LT14=sum(LT2014[LT2014.Feeder==feedername].iloc[:,1])
    if feedername in LTList2013:
        LT13=sum(LT2013[LT2013.Feeder==feedername].iloc[:,1])
    if feedername in LTList2012:
        LT12=sum(LT2012[LT2012.Feeder==feedername].iloc[:,1])

    if feedername in AddList2017:
        if feedername[0]=="8":
            Add17=sum(Add2017[Add2017.Feeder==feedername].iloc[:,1])/(13.8*1.732)
        if feedername[0]=="2":
            Add17=sum(Add2017[Add2017.Feeder==feedername].iloc[:,1])/(26*1.732)
    if feedername in AddList2016:
        if feedername[0]=="8":
            Add16=sum(Add2016[Add2016.Feeder==feedername].iloc[:,1])/(13.8*1.732)
        if feedername[0]=="2":
            Add16=sum(Add2016[Add2016.Feeder==feedername].iloc[:,1])/(26*1.732)
    if feedername in AddList2015:
        if feedername[0]=="8":
            Add15=sum(Add2015[Add2015.Feeder==feedername].iloc[:,1])/(13.8*1.732)
        if feedername[0]=="2":
            Add15=sum(Add2015[Add2015.Feeder==feedername].iloc[:,1])/(26*1.732)
    if feedername in AddList2014:
        if feedername[0]=="8":
            Add14=sum(Add2014[Add2014.Feeder==feedername].iloc[:,1])/(13.8*1.732)
        if feedername[0]=="2":
            Add14=sum(Add2014[Add2014.Feeder==feedername].iloc[:,1])/(26*1.732)
    if feedername in AddList2013:
        if feedername[0]=="8":
            Add13=sum(Add2013[Add2013.Feeder==feedername].iloc[:,1])/(13.8*1.732)
        if feedername[0]=="2":   
            Add13=sum(Add2013[Add2013.Feeder==feedername].iloc[:,1])/(26*1.732)
    if feedername in AddList2012:
        if feedername[0]=="8":
            Add12=sum(Add2012[Add2012.Feeder==feedername].iloc[:,1])/(13.8*1.732)
        if feedername[0]=="2":
            Add12=sum(Add2012[Add2012.Feeder==feedername].iloc[:,1])/(26*1.732)
    if feedername in AddList2011:
        if feedername[0]=="8":
            Add11=sum(Add2011[Add2011.Feeder==feedername].iloc[:,1])/(13.8*1.732)
        if feedername[0]=="2":
            Add11=sum(Add2011[Add2011.Feeder==feedername].iloc[:,1])/(26*1.732)
    if feedername in AddList2010:
        if feedername[0]=="8":
            Add10=sum(Add2010[Add2010.Feeder==feedername].iloc[:,1])/(13.8*1.732)
        if feedername[0]=="2":
            Add10=sum(Add2010[Add2010.Feeder==feedername].iloc[:,1])/(26*1.732)
    if feedername in AddList2009:
        if feedername[0]=="8":
            Add09=sum(Add2009[Add2009.Feeder==feedername].iloc[:,1])/(13.8*1.732)
        if feedername[0]=="2":    
            Add09=sum(Add2009[Add2009.Feeder==feedername].iloc[:,1])/(26*1.732)
    
    for k in range(0,9):
        row4=array(data.iloc[i,3+k:3+k+4])
        if k==0:
            GDP=[14.2,9.1,-2.5]
            MaxT=[32.3,34,33.2]
            T=[-0.2,1.7,-0.8]
            ResRatio=[ResRatio2009,ResRatio2009,ResRatio2009]
            LT=[0,0,0]
            Add=[0,0,Add09]
        if k==1:
            GDP=[9.1,-2.5,2.2]
            MaxT=[34,33.2,31.4]
            T=[1.7,-0.8,-1.8]
            ResRatio=[ResRatio2009,ResRatio2009,ResRatio2010]
            LT=[0,0,0]
            Add=[0,Add09,Add10]
        if k==2:
            GDP=[-2.5,2.2,3.2]
            MaxT=[33.2,31.4,30.4]
            T=[-0.8,-1.8,-1]
            ResRatio=[ResRatio2009,ResRatio2010,ResRatio2011]
            LT=[0,0,0]
            Add=[Add09,Add10,Add11]
        if k==3:
            GDP=[2.2,3.2,3.5]
            MaxT=[31.4,30.4,30.5]
            T=[-1.8,-1,0.1]
            ResRatio=[ResRatio2010,ResRatio2011,ResRatio2012]
            LT=[0,0,LT12]
            Add=[Add10,Add11,Add12]
        if k==4:
            GDP=[3.2,3.5,2.26]
            MaxT=[30.4,30.5,32.8]
            T=[-1,0.1,2.3]
            ResRatio=[ResRatio2011,ResRatio2012,ResRatio2013]
            LT=[0,LT12,LT13]
            Add=[Add11,Add12,Add13]
        if k==5:
            GDP=[3.5,2.26,3.9]
            MaxT=[30.5,32.8,32.2]
            T=[0.1,2.3,-0.6]
            ResRatio=[ResRatio2012,ResRatio2013,ResRatio2014]
            LT=[LT12,LT13,LT14]
            Add=[Add12,Add13,Add14]
        if k==6:
            GDP=[2.26,3.9,-0.2]
            MaxT=[32.8,32.2,33.6]
            T=[2.3,-0.6,1.4]
            ResRatio=[ResRatio2013,ResRatio2014,ResRatio2015]
            LT=[LT13,LT14,LT15]
            Add=[Add13,Add14,Add15]            
        if k==7:
            GDP=[3.9,-0.2,-3.16]
            MaxT=[32.2,33.6,30.9]
            T=[-0.6,1.4,-2.7]
            ResRatio=[ResRatio2014,ResRatio2015,ResRatio2016]
            LT=[LT14,LT15,LT16]
            Add=[Add14,Add15,Add16]
        if k==8:
            GDP=[-0.2,-3.16,3]
            MaxT=[33.6,30.9,33]
            T=[1.4,-2.7,2.1]
            ResRatio=[ResRatio2015,ResRatio2016,ResRatio2016]
            LT=[LT15,LT16,LT17]
            Add=[Add15,Add16,Add17]
            
        if row4[3]>0 and row4[2]>0 and row4[1]>0 and row4[0]>0:            
            #if (abs(row4[3]-row4[2])/row4[2]<0.2 and abs(row4[2]-row4[1])/row4[1]<0.2 and abs(row4[1]-row4[0])/row4[0]<0.2) or ResRatio<0.7:
            if (abs(row4[3]-row4[2])/row4[2]<0.2 and abs(row4[2]-row4[1])/row4[1]<0.2 and abs(row4[1]-row4[0])/row4[0]<0.2) and k<5:
                X=X.append({'Load':row4[0],'ResRatio':ResRatio[0],'GDP':GDP[0],'MaxT':MaxT[0],'T':T[0],'LT':LT[0],'Add':Add[0]},ignore_index=True)
                Y=Y.append({'NLoad':row4[1]},ignore_index=True)
                X=X.append({'Load':row4[1],'ResRatio':ResRatio[1],'GDP':GDP[1],'MaxT':MaxT[1],'T':T[1],'LT':LT[1],'Add':Add[1]},ignore_index=True)
                Y=Y.append({'NLoad':row4[2]},ignore_index=True)
                X=X.append({'Load':row4[2],'ResRatio':ResRatio[2],'GDP':GDP[2],'MaxT':MaxT[2],'T':T[2],'LT':LT[2],'Add':Add[2]},ignore_index=True)
                Y=Y.append({'NLoad':row4[3]},ignore_index=True)
            if   k>=5:
                X=X.append({'Load':row4[0],'ResRatio':ResRatio[0],'GDP':GDP[0],'MaxT':MaxT[0],'T':T[0],'LT':LT[0],'Add':Add[0]},ignore_index=True)
                Y=Y.append({'NLoad':row4[1]},ignore_index=True)
                X=X.append({'Load':row4[1],'ResRatio':ResRatio[1],'GDP':GDP[1],'MaxT':MaxT[1],'T':T[1],'LT':LT[1],'Add':Add[1]},ignore_index=True)
                Y=Y.append({'NLoad':row4[2]},ignore_index=True)
                X=X.append({'Load':row4[2],'ResRatio':ResRatio[2],'GDP':GDP[2],'MaxT':MaxT[2],'T':T[2],'LT':LT[2],'Add':Add[2]},ignore_index=True)
                Y=Y.append({'NLoad':row4[3]},ignore_index=True)     
                
Z1=pd.concat([X,Y], axis=1) # directly combine X and Y horizontally 

#Z=Z1.drop_duplicates (subset="Load") # drop duplicates by column load
Z=Z1.drop_duplicates () # drop duplicates by column load

# Part 2A - Making the ANN with single year

X=Z[['Load','ResRatio','MaxT','T','LT','Add']]
y=Z[['NLoad']]
                
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = MinMaxScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu'))
#regressor.add(Dense(6))
#regressor.add(Dropout(0.1))

## Adding the second hidden layer
#regressor.add(Dense(units = 58, kernel_initializer = 'he_uniform', activation = 'selu'))
##regressor.add(Dropout(0.1))

# Adding the output layer
regressor.add(Dense(units = 1, activation='relu'))

# Compiling the ANN
#from keras.optimizers import Adam
#optimizer = Adam(lr=0.3)
regressor.compile(loss='mae', optimizer='adam',metrics = ['mean_absolute_percentage_error'])

# Fitting the ANN to the Training set
history = regressor.fit(X_train, y_train, batch_size =12, epochs = 200)
# Plot Metrics
from matplotlib import pyplot
pyplot.plot(history.history['mean_absolute_percentage_error'])
#pyplot.plot(history.history['Loss'])
pyplot.show()                

y_pred = regressor.predict(X_test)
#y_pred = (y_pred > 0.5)
y_predreal=sc_y.inverse_transform(y_pred)
y_testreal=sc_y.inverse_transform(y_test)
np.mean(abs(y_testreal-y_predreal)/y_testreal)

# Part 2B - Making the ANN with every three years

X=Z1[['Load','GDP','MaxT','T','ResRatio','LT','Add']]
t=7
y=Z1[['NLoad']]

XL=array(X).reshape(int(len(X)/3),3*t)
YL=array(Y).reshape(int(len(X)/3),3)
YLOne=YL[:,2] #Obtain only the last column as the output

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(XL, YLOne, test_size = 0.2)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = MinMaxScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
y_test = sc_y.transform(y_test.reshape(-1,1))

# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(units = 8, kernel_initializer = 'glorot_uniform', activation = 'relu'))

# Adding the output layer
regressor.add(Dense(units = 1, activation='relu'))

# Compiling the ANN
#from keras.optimizers import Adam
#optimizer = Adam(lr=0.3)
regressor.compile(loss='mae', optimizer='adam',metrics = ['mean_absolute_percentage_error'])

# Fitting the ANN to the Training set
history = regressor.fit(X_train, y_train, batch_size =12, epochs = 200)
# Plot Metrics
from matplotlib import pyplot
pyplot.plot(history.history['mean_absolute_percentage_error'])
#pyplot.plot(history.history['Loss'])
pyplot.show()                

y_pred = regressor.predict(X_test)
#y_pred = (y_pred > 0.5)
y_predreal=sc_y.inverse_transform(y_pred)
y_testreal=sc_y.inverse_transform(y_test)
print (np.mean(abs(y_testreal-y_predreal)/y_testreal))
print (np.max(abs(y_testreal-y_predreal)/y_testreal))

 #Part 3 - Making the LSTM - Many to Many with Examples on Split Shifted Many-to-Many, TimeDistributed Output

# Use the orignal Z (three rows for every timestep). There are repeated ones but doesn't matter for training.
X=Z1[['Load','GDP','MaxT','T','ResRatio','LT','Add']]
t=7
y=Z1[['NLoad']]

xl=array(X).reshape(int(len(X)/3),3*t)
yl=array(Y).reshape(int(len(X)/3),3)

#np.random.seed(16)
from sklearn.model_selection import train_test_split
avgmape=1
while avgmape>0.07:
    X_train, X_test, y_train_R, y_test_R = train_test_split(xl, yl, test_size = 0.2)
    
    # Feature Scaling
    #from sklearn.preprocessing import StandardScaler
    y_train=y_train_R
    y_test=y_test_R
    from sklearn.preprocessing import MinMaxScaler
    sc_X = MinMaxScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    sc_y = MinMaxScaler()
    y_train = sc_y.fit_transform(y_train)
    y_test = sc_y.transform(y_test)
    y_train2yr=y_train[:,0:2] # This is for Split Shifted Many-to-Many(only forecasting 2 years in the next 3 years)
    
    X_train3D=X_train.reshape(X_train.shape[0],3,t)
    X_test3D=X_test.reshape(X_test.shape[0],3,t)
    y_train3D=y_train.reshape(y_train.shape[0],3,1)
    
    # Initialising the RNN
    start=time.time()
    regressor = Sequential()
    
    # Adding the input layer and the first hidden layer
    #regressor.add(LSTM(18, input_shape=(X_train3D.shape[1], X_train3D.shape[2]),kernel_initializer = 'he_uniform', activation = 'selu'))
    #regressor.add(LSTM(16, input_shape=(X_train3D.shape[1], X_train3D.shape[2])))
#   regressor.add(SimpleRNN(16, input_shape=(X_train3D.shape[1], X_train3D.shape[2])))
    regressor.add(SimpleRNN(16, input_shape=(X_train3D.shape[1], X_train3D.shape[2]),return_sequences=True))
    
    # Adding the output layer
    #regressor.add(Dense(units = 3, kernel_initializer = 'he_uniform', activation = 'selu'))
#    regressor.add(Dense(2,activation='relu'))
    from keras.layers import TimeDistributed #Use TimeDistributed to Forecast to output at every timestep
    regressor.add(TimeDistributed(Dense(1,activation='relu')))
    
    # Compiling the ANN
    #from keras.optimizers import Adam
    #optimizer = Adam(lr=0.3)
    regressor.compile(loss='mae', optimizer='adam',metrics = ['mean_absolute_percentage_error'])
    
    # Fitting the ANN to the Training set
#    history = regressor.fit(X_train3D, y_train, batch_size=12, epochs = 200)
    history = regressor.fit(X_train3D, y_train3D, batch_size=12, epochs = 200) #With TimeWrapper
#    history = regressor.fit(X_train3D, y_train2yr, batch_size=12, epochs = 200) #Split Year: Only forecast  2 years in future
    # Plot Metrics
    from matplotlib import pyplot
    #pyplot.plot(history.history['mean_absolute_percentage_error'])
    #pyplot.plot(history.history['Loss'])
    #pyplot.show()                
    
    y_pred = regressor.predict(X_test3D)
    #y_pred = (y_pred > 0.5)
    y_predreal=sc_y.inverse_transform(y_pred)
    y_testreal=sc_y.inverse_transform(y_test)
    print (np.mean(abs(y_testreal[:,2]-y_predreal[:,2])/y_testreal[:,2]))
    print (np.max(abs(y_testreal[:,2]-y_predreal[:,2])/y_testreal[:,2]))
    print (time.time()-start)
    mape=array(abs(y_testreal[:,2]-y_predreal[:,2])/y_testreal[:,2]*100)
    avgmape=np.mean(abs(y_testreal[:,2]-y_predreal[:,2])/y_testreal[:,2])
#Part 3 - Making the LSTM - Many to One

# Use the orignal Z (three rows for every timestep). There are repeated ones but doesn't matter for training.
X=Z1[['Load','GDP','MaxT','T','ResRatio','LT','Add']]
t=7
y=Z1[['NLoad']]

xl=array(X).reshape(int(len(X)/3),3*t)
yl=array(Y).reshape(int(len(X)/3),3)
YLOne=yl[:,2] #Obtain only the last column as the output

#np.random.seed(16)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(xl, YLOne, test_size = 0.2)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
y_train=y_train_R[:,2]
y_test=y_test_R[:,2]

from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = MinMaxScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1)) 
y_test = sc_y.transform(y_test.reshape(-1,1))

X_train3D=X_train.reshape(X_train.shape[0],3,t)
X_test3D=X_test.reshape(X_test.shape[0],3,t)

# Initialising the RNN
#np.random.seed(6)
start=time.time()
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(SimpleRNN(units=16, input_shape=(X_train3D.shape[1], X_train3D.shape[2])))
#regressor.add(GRU(16, input_shape=(X_train3D.shape[1], X_train3D.shape[2])))

# Adding the output layer
#regressor.add(Dense(units = 1, kernel_initializer = 'he_uniform', activation = 'selu'))
regressor.add(Dense(units = 1))

# Compiling the ANN
#optimizer1 = Adam(lr=0.001) #Adam optimizer's default learning rate is 0.001
regressor.compile(loss='mae', optimizer='adam',metrics = ['mean_squared_error'])

# Fitting the ANN to the Training set
history = regressor.fit(X_train3D, y_train, batch_size=12, epochs = 200)
# Plot Metrics
from matplotlib import pyplot
pyplot.plot(history.history['mean_squared_error'])
#pyplot.plot(history.history['Loss'])
pyplot.show()                

y_pred = regressor.predict(X_test3D)
#y_pred = (y_pred > 0.5)
y_predreal=sc_y.inverse_transform(y_pred)
y_testreal=sc_y.inverse_transform(y_test)
#y_predreal=y_pred #When Y is not normalized
#y_testreal=y_test #When Y is not normalized

print (np.mean(abs(y_testreal-y_predreal)/y_testreal))
print (np.max(abs(y_testreal-y_predreal)/y_testreal))
print (time.time()-start)
mape=array(abs(y_testreal-y_predreal)/y_testreal*100)

# Save Trained Model and Weights
regressor.save_weights('model_weights.h5')
with open('model_architecture.json', 'w') as f:
    f.write(regressor.to_json())
    
#Read Trained Model and Weights   
#from keras.models import model_from_json
#      
#with open('model_architecture.json', 'r') as f:
#     regressor = model_from_json(f.read())
#regressor.load_weights('model_weights.h5')
    
#Std Calculation
    std=0
    for i in range (289):
       std=std+np.std(y_test_R[i])
     
    std=std/i
    print(std)