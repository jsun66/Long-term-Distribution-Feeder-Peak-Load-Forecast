# Background and Purpose
In power systems, distribution feeder long-term load forecast is a critical task many electric utility companies perform on an annual basis. The goal of this project is to forecast the load change on existing distribution feeders for the next few years. The forecasted results will be used as input in long-term system planning studies to determine necessary system upgrades so that the distribution system can continue to operate reliably during normal operation and contingences.
# Modeling Method
A comprehensive hybrid model based on LSTM neural network: It is not only able to combine the advantages of top-down and bottom-up forecasting models but also able to leverage the time-series characteristics of multi-year data.

The structure of the modeling method is shown in Fig.1. Raw top-down features related to economy, population and temperature, raw bottom-up features related to large customers, DER and EV adoption, previous feeder peak demand are all fed into the feature engineering module. For feature engineering, the concept of virtual feeder features is used to eliminate the data noise resulted from historical load transfer events between feeders; principle component analysis is applied to reduce the dimensionality of highly correlated features to improve learning efficiency and avoid over-fitting problems; then feature normalization is applied to normalize different types of features to the same numerical scale. After the step of feature engineering, the dataset is constructed to a unique multi-step format to be compatible with LSTM neural network. The dataset is also split into training set and test set for training and evaluation of the LSTM model. In the end, a reliable LSTM model for distribution feeder long-term peak demand forecast is established and ready to be used for forecast.

![Alt text](https://github.com/jsun66/Long-term-Distribution-Feeder-Peak-Load-Forecast/blob/main/Tables%20and%20Figures/Fig.1.%20Workflow%20of%20the%20modeling%20method.png)
Fig.1. Workflow of the modeling method

LSTM neural network serves as the foundation of the load forecasting mathematical model. It is an enhanced Recurrent Neural Network (RNN) that solves the vanishing/exploding gradient problem of a traditional RNN. Compared to traditional RNN, LSTM introduces a specially designed LSTM unit to sophisticatedly control the flow of hidden state information from one time step to the next. The structure of LSTM unit is shown in Fig.2.

![Alt text](https://github.com/jsun66/Long-term-Distribution-Feeder-Peak-Load-Forecast/blob/main/Tables%20and%20Figures/Fig.2.%20A%20LSTM%20unit%20diagram.png)
Fig.2. A LSTM unit diagram

LSTM neural network inherits the advantages of RNN in dealing with temporal forecast problems and also solved the vanishing/exploding gradient problem. It is therefore chosen as the ideal mathematical model for long-term peak demand forecast in this research.
# Results
The method was applied to a large urban grid in West Canada to establish both summer and winter long-term peak demand forecasting models for its distribution feeders that are serving various types of loads. In total 289 distribution feeders were selected and their past 14-year annual data were used to create the dataset. In total 1,997 valid three-year records were produced for both summer and winter. In order to reveal the true forecasting capability, for each year, instead of using the actual values, forecasted economic and population features prior to that year were used. The 1,997 records were split into 1,597 records for training and 400 records for testing based on the 80%/20% split ratio. To evaluate the model’s forecast accuracy; the trained model was tested on the 400 test records and compared to the true peak demand values. Mean Absolute Percentage Error (MAPE) is chosen as the error metrics.

1. MAPE in Summer and Winter

The results show that MAPE in summer is 6.77% and 4.87% in winter. The histograms and cumulative percentages for both seasons are plotted in Fig.3. 84.08% of winter forecasts have less than 10% MAPE and 86.00% of summer forecasts have less than 10% MAPE. After investigation, it was found that most large errors are attributed to abnormal load behaviors during two dramatic economic downturns in 2009 and 2015-2016 in this region. Overall, the results are quite accurate.

![Alt text](https://github.com/jsun66/Long-term-Distribution-Feeder-Peak-Load-Forecast/blob/main/Tables%20and%20Figures/Fig.3(a).%20MAPE%20in%20summer.png)

Fig.3(a) MAPE in summer

![Alt text](https://github.com/jsun66/Long-term-Distribution-Feeder-Peak-Load-Forecast/blob/main/Tables%20and%20Figures/Fig.3(b).%20MAPE%20in%20winter.png)

Fig.3(b) MAPE in winter

2. Comparison to Other Models

As part of the model evaluation, the model was compared to various other models and their performance is summarized in the below table:

![Alt text](https://github.com/jsun66/Long-term-Distribution-Feeder-Peak-Load-Forecast/blob/main/Tables%20and%20Figures/Table%201.PNG)

As shown in Table I, the proposed model outperformed all other models in both summer and winter forecasting.
