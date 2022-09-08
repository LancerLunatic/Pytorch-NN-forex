#region imports
from AlgorithmImports import *
#endregion
import clr
clr.AddReference("System")
clr.AddReference("QuantConnect.Algorithm")
clr.AddReference("QuantConnect.Common")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *

import numpy as np
import torch
import torch.nn.functional as F

class PytorchNeuralNetworkAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2017, 3, 1)  # Set Start Date
        #self.SetEndDate(2021, 1, 8) # Set End Date
        self.SetWarmup(7)
        self.SetCash(100000)  # Set Strategy Cash
        self.SetBrokerageModel(BrokerageName.OandaBrokerage)
        # add symbol
        spy = self.AddForex("EURUSD", Resolution.Minute, Market.Oanda)
        spy1 = self.AddForex("USDCHF", Resolution.Minute, Market.Oanda)
        spy2 = self.AddForex("GBPUSD", Resolution.Minute, Market.Oanda)
        spy3 = self.AddForex("EURGBP", Resolution.Minute, Market.Oanda)
        #spy4 = self.AddForex("EURCHF", Resolution.Minute, Market.Oanda)
        #spy5 = self.AddForex("GBPCHF", Resolution.Minute, Market.Oanda)
        #spy6 = self.AddForex("AUDCAD", Resolution.Minute, Market.Oanda)
        #spy7 = self.AddForex("AUDNZD", Resolution.Minute, Market.Oanda)
        
        self.symbols = [spy.Symbol, spy1.Symbol, spy2.Symbol, spy3.Symbol]#, spy4.Symbol, spy5.Symbol, spy6.Symbol, spy7.Symbol] # using a list can extend to condition for multiple symbols
        
        self.lookback = 30 # days of historical data (look back)
        self.SetBenchmark("EURUSD")
        self.Schedule.On(self.DateRules.WeekStart("EURUSD"), self.TimeRules.AfterMarketOpen("EURUSD"), self.NetTrain) # train the NN
        self.Schedule.On(self.DateRules.WeekStart("EURUSD"), self.TimeRules.AfterMarketOpen("EURUSD"), self.Trade)
        #self.Schedule.On(self.DateRules.EveryDay("EURUSD"), self.TimeRules.Every(timedelta(minutes=1420)), self.NetTrain) # train the NN
        #self.Schedule.On(self.DateRules.EveryDay("EURUSD"), self.TimeRules.Every(timedelta(minutes=1440)), self.Trade)
    def OnData(self, data):
        if self.IsWarmingUp: return
        if (not data.ContainsKey(self.symbols[0])): return
        if (not data.ContainsKey(self.symbols[1])): return
    def NetTrain(self):
        # Daily historical data is used to train the machine learning model
        history = self.History(self.symbols, self.lookback + 1, Resolution.Daily)
        
        # dicts that store prices for training
        self.prices_x = {} 
        self.prices_y = {}
        
        # dicts that store prices for sell and buy
        self.sell_prices = {}
        self.buy_prices = {}
        
        for symbol in self.symbols:
            if not history.empty:
                # x: preditors; y: response
                self.prices_x[symbol] = list(history.loc[symbol.Value]['open'])[:-1]
                self.prices_y[symbol] = list(history.loc[symbol.Value]['open'])[1:]
                
        for symbol in self.symbols:
            # if this symbol has historical data
            if symbol in self.prices_x:
                
                net = Net(n_feature=1, n_hidden=75, n_output=1)     # define the network
                optimizer = torch.optim.SGD(net.parameters(), lr=0.04)
                loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
                
                for t in range(90):
                    # Get data and do preprocessing
                    x = torch.from_numpy(np.array(self.prices_x[symbol])).float()
                    y = torch.from_numpy(np.array(self.prices_y[symbol])).float()
                    
                    # unsqueeze data (see pytorch doc for details)
                    x = x.unsqueeze(1) 
                    y = y.unsqueeze(1)
                
                    prediction = net(x)     # input x and predict based on x

                    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

                    optimizer.zero_grad()   # clear gradients for next train
                    loss.backward()         # backpropagation, compute gradients
                    optimizer.step()        # apply gradients
            
            # Follow the trend    
            self.buy_prices[symbol] = net(y)[-1] + np.std(y.data.numpy())
            self.sell_prices[symbol] = net(y)[-1] - np.std(y.data.numpy())
        
    def Trade(self):
        ''' 
        Enter or exit positions based on relationship of the open price of the current bar and the prices defined by the machine learning model.
        Liquidate if the open price is below the sell price and buy if the open price is above the buy price 
        ''' 
        #if self.modelIsTraining:
        #    return
        
        for symbol in self.symbols:
            price =self.Securities[symbol].Price
        for symbol in self.symbols:
            if price < self.sell_prices[symbol]:#
                self.SetHoldings(symbol, 1)
                #self.Liquidate(symbol, "liquidate shorts except AUDCAD & audchf")    
            elif price > self.buy_prices[symbol]:#
                #self.Liquidate(symbol, "Liquidate all shorts except audcad")
                self.SetHoldings(symbol, -1)
# class for Pytorch NN model
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
    
    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
