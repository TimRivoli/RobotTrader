import datetime, os, itertools, multiprocessing, pandas as pd, numpy as np
from datetime import datetime
from os import listdir
from _classes.PriceTradeAnalyzer import TradingModel, ForcastModel, PricingData, PriceSnapshot
from _classes.Utility import *
from multiprocessing import Pool
from random import randint

#This will do one trade per day, with a goal of picking the most profitable action.  Actions are pooled into sequences, most every permutation is forecast for a period of days to see which is most profitable
#The model will be trained on the best possible actions forecast and then asked to make its own predictions.  CNN or LSTM models can be used.
OrderDuration = 3 		#for non-market orders, how many days until canceled
TradeSequenceLength = 4	#number of sequential actions evaluated, evaluations get exponentially more complex so over 7 is going to get real slow
ForcastDuration = 20 	#After executing sequence, how many days to project before evaluating the effect
Tranches = 30			#Chunks of funds to be used for purchases
TranchSize = 1000		#Dollar amount.  Make this enough to buy at least 10 shares of whatever ticker you choose
InitialFunds = TranchSize * Tranches
BestActionDataFolder = 'data/bestactions/'	#Where the training data will be stored
WindowSize = 60 			#When using a CCN model this is the input window size
UseLSTM = False 			#We can use LSTM or CNN
if UseLSTM: WindowSize = 1	#Historical windows are implicit with LSTM
train_test_split = .8 		#Split between data to be trained on and data to be evaluated upon.  Global variable so this can also be used to compare against BuyHold dates

#---------------------------------------- Global Helpers -------------------------------------------------
def TestStartDate(startDate, endDate):
	startDate = ToDate(startDate)
	endDate = ToDate(endDate)
	totalDuration = endDate-startDate
	daysUntilTest = int(totalDuration.days * train_test_split)
	r = AddDays(startDate, daysUntilTest) 
	return r

def RecordPerformance(ModelName, StartDate, TestStartDate, EndDate, StartValue, TestStartValue, EndValue, TradeCount):
	#Record trade performance to output file, each model run will append to the same files so you can easily compare the results, the result of training period and testing period are separately broken out
	filename = 'data/trademodel/RTPerfomanceComparisons.csv'
	try:
		if FileExists(filename):
			f = open(filename,"a")
		else:
			f = open(filename,"w+")
			f.write('ModelName, StartDate, TestStartDate, EndDate, StartValue, TestStartValue, EndValue, TotalPercentageGain, TestPercentageGain, TradeCount\n')
		TotalPercentageGain = (EndValue/StartValue)-1
		TestPercentageGain = (EndValue/TestStartValue)-1
		f.write(ModelName + ',' + str(StartDate) + ',' + str(TestStartDate) + ',' + str(EndDate) + ',' + str(StartValue) + ',' + str(TestStartValue) + ',' + str(EndValue) + ',' + str(TotalPercentageGain) + ',' + str(TestPercentageGain) + ',' + str(TradeCount) + '\n')
		f.close() 
	except:
		print('Unable to write performance report to ' + filename)

def ModelName(ticker:str): return 'RT_' + ticker + '_seqlen' + str(TradeSequenceLength) + '_forcast' + str(ForcastDuration) + '_FundSets' + str(Tranches)

def MakeState(mode:int, p:PriceSnapshot, available:float, buys:float=0, sells:float=0, long:float=0):
	#returns an array of numbers to describe current market state of given day, used as features for the model input, experiment freely here but I've found simple is better.  This is how the model sees the state of the world.
	#high, low, open, close, oneDayAverage, twoDayAverage, shortEMA, shortEMASlope, longEMA, longEMASlope, channelHigh, channelLow, oneDayApc, oneDayDeviation, fiveDayDeviation, fifteenDayDeviation
	#estLow, nextDayTarget, estHigh are all available fields
	#The current state of the portfolio (available funds, open orders, long positions) are also part of the input
	size = 0
	r = []
	try:
		percentageAvailable = 0
		if available > 0: percentageAvailable = available / (available + buys + sells + long)
	except:
		percentageAvailable = 0
	if mode==0:	
		size = 3
		if p != None: r = [p.oneDayAverage, p.fiveDayDeviation, percentageAvailable]	
	elif mode==1: #Does best on the test set
		size = 3
		if p != None: r = [p.oneDayApc, p.fiveDayDeviation, percentageAvailable]	
	elif mode==2:
		size = 4
		if p != None: r = [p.low, p.oneDayAverage, p.high, percentageAvailable]
	elif mode==3:	#388 epochs, 98% CNN Win30. Best overall so far
		size = 6
		if p != None: r = [p.low, p.oneDayAverage, p.high, p.shortEMA, p.longEMA, percentageAvailable]	
	elif mode==4:	
		size = 1
		if p != None: r = [p.oneDayAverage]	
	elif mode==5:	#350 epochs, 57% CNN Win30. Learns no more with available, buys, sells, long added
		size = 3
		if p != None: r = [p.low, p.oneDayAverage, p.high]	
	elif mode==6: 
		size = 5
		if p != None: r = [p.oneDayApc, p.fiveDayDeviation, p.low, p.oneDayAverage, p.high]	
	if p == None:
		for _ in range(size):
			r += [0]
	return r
	
def DoAction(tm, p:PriceSnapshot, actionID:int, verbose:bool=False):
	#Translates a numeric actionID until a specific trade for the TradeModel to perform
	available, buys, sells, long = tm.PositionSummary()
	if available > 0:
		if (actionID==0):	#BuyAggressive
			tm.PlaceBuy(ticker=p.ticker, price=p.nextDayTarget*(1-p.fiveDayDeviation/2), marketOrder=False, expireAfterDays=OrderDuration, verbose=verbose)
		if (actionID==1):	#BuyTarget
			tm.PlaceBuy(ticker=p.ticker, price=p.nextDayTarget*(1-p.fiveDayDeviation/4), marketOrder=False, expireAfterDays=OrderDuration, verbose=verbose)
		if (actionID==2):	#BuyMarket
			tm.PlaceBuy(ticker=p.ticker, price=p.nextDayTarget, marketOrder=True, expireAfterDays=OrderDuration, verbose=verbose)
	if (actionID==3): 	#Hold
		pass
	if long > 0:
		if (actionID==4):	#SellMarket
			tm.PlaceSell(ticker=p.ticker, price=p.nextDayTarget, marketOrder=True, expireAfterDays=OrderDuration, verbose=verbose)
		if (actionID==5):	#SellTarget
			tm.PlaceSell(ticker=p.ticker, price=p.nextDayTarget*(1+p.fiveDayDeviation/4), marketOrder=False, expireAfterDays=OrderDuration, verbose=verbose)
		if (actionID==6):	#SellAgressive
			tm.PlaceSell(ticker=p.ticker, price=p.nextDayTarget*(1+p.fiveDayDeviation/2), marketOrder=False, expireAfterDays=OrderDuration, verbose=verbose)
	if (actionID==7):	#CancelAllOrders, presumably buys+sells > 0
		tm.CancelAllOrders()
		
#------------------------------------ Best action calculation for supervised learning ----------------------------------------------------

def BestActionFileName(ticker:str, startDate:str=None): 
	#Helper function to create a best action file name for a given ticker and model
	r = BestActionDataFolder + ModelName(ticker) + '.csv'
	if startDate is not None: r = r[:-4] + '_' + startDate[-4:] + '.csv'
	return r

def SweepBestActionFiles(ticker):
	#Combine best action files having the same root name into one larger file, this allows you to combine files from different date ranges
	baFile = BestActionFileName(ticker)
	rootPath = baFile[:-4] + '_' 
	x = None
	for f in listdir(BestActionDataFolder):
		fp = BestActionDataFolder + f
		if fp[:len(rootPath)] == rootPath:
			print(fp)
			xx = pd.read_csv(fp, index_col=0, parse_dates=True, na_values=['nan'])
			if x is None:
				x = xx
			else:
				x = x.append(xx)	
	if x is not None:
		x.sort_index(inplace=True)	
		x.to_csv(baFile)
		print('Best actions swept into ' + baFile)
				
def ExecuteSequence(tm, s, verbose:bool=False):
	for ii in range(len(s)):
		p = tm.GetPriceSnapshot()
		DoAction(tm, p, s[ii], verbose)
		tm.ProcessDay()

def ForcastSequence(fm, s, verbose:bool=False):
	fm.Reset(True)
	ExecuteSequence(fm.tm, s, verbose)
	return fm.GetResult()

def OptimizeBestActions(ticker:str, tradeSequenceLen:int=TradeSequenceLength):	
	#I put a little more effort into optimizing bestAction[0] so this is to propagate that to prior days
	filePath = BestActionFileName(ticker)
	bestActions = pd.read_csv(filePath, index_col=0, parse_dates=True, na_values=['nan'])
	betterActions = bestActions.copy()
	betterActions.replace(to_replace=-1, value=4, inplace=True) #Bad month, sell.  Also, any 7s would indicate bad prior decisions
	for i in range(tradeSequenceLen):
		betterActions[str(i+1)] = betterActions[str(i)].shift(-1).fillna(0.0).astype(int)
	betterActions.iloc[-tradeSequenceLen:] = bestActions.iloc[-tradeSequenceLen:]
	#print(bestActions.join(betterActions, how='outer', rsuffix='_b'))
	betterActions.to_csv(filePath + '_optimized.csv')
	print('Actions have been optimized.')
			
def BestSequence(fm: ForcastModel, tradeSequences, verbose:bool=False):
	#Calculates the most profitable sequence of trades using a forecast model
	bestForcastResult = -10000	#A really bad month, where no sequence will save us for losing this amount.  Should be a string of market sells 
	bestForcastSequence = [-1]*TradeSequenceLength
	for i in range(len(tradeSequences)):
		s = tradeSequences[i]
		r = ForcastSequence(fm, s, False)
		if r == bestForcastResult: 
			if s.count(3) > bestForcastSequence.count(3) or s[0]==3: #two equivalent sequences pick the one with the most holds
				bestForcastResult = r
				bestForcastSequence = s
				if verbose: print(' Equivalent sequence ', s, ' has fewer actions')
		if r > bestForcastResult: 
			bestForcastResult = r
			bestForcastSequence = s
			if verbose: print(' Sequence ', s, ' results in ', r, ' after ', ForcastDuration, ' days')
	return bestForcastResult, bestForcastSequence
	
def BetterSequence(fm: ForcastModel, bestForcastResult, bestForcastSequence):
	#Tries to find a more profitable sequence of trades by placing more aggressive orders
	secondPassSequence = [[0,],[1,],[2,]]
	if bestForcastSequence[0] == 1 or bestForcastSequence[0] == 5 or bestForcastResult < 0:
		secondPassSequence[0] = list(bestForcastSequence)
		secondPassSequence[1] = list(bestForcastSequence)
		secondPassSequence[2] = list(bestForcastSequence)	
		if bestForcastSequence[0] == 1:  #Try more aggressive buy
			secondPassSequence[1][0] = 0
			secondPassSequence[2][0] = 2
		if bestForcastSequence[0] == 5 or bestForcastResult < 0:	#Try more aggressive sell
			secondPassSequence[1][0] = 4
			secondPassSequence[2][0] = 6
		bestForcastResult, bestForcastSequence = BestSequence(fm, secondPassSequence)
	return bestForcastResult, bestForcastSequence

def CalculateBestActions(ticker:str, startDate:str, durationInYears:int, plotResults:bool=False, saveHistoryToFile:bool=False, returndailyValues:bool=False):
	#Calculate the most profitable trade sequences possible for a given period, results are saved to .csv and used for training
	#Results are dependent on the ticker, sequence length, forecast duration, and total funds.  Changing any of those requires a new file that will have a different file name.  
	#Calculations with the same variables on different date periods can be combined together with SweepBestActionFiles to avoid recalculating
	if not CreateFolder(BestActionDataFolder):
		print('Unable to create folder: ', BestActionDataFolder)
		assert(False)
	ThreadCount = 3
	#tradeSequences = [_ for _ in itertools.product([1,3,5,7], repeat=TradeSequenceLength)] #3**6 = 729; 4**6 = 4096; options to test per day.  plus 2.
	x = [1,3,5,7]
	tradeSequences1 = list(itertools.product([1],x,x,x))
	tradeSequences3 = list(itertools.product([3],x,x,x))
	tradeSequences5 = list(itertools.product([5],x,x,x))
	tsPool = Pool(ThreadCount)
	pooledResults = []
	modelName = ModelName(ticker)+ '_BestAction'
	tm = TradingModel(modelName=modelName, startingTicker=ticker, startDate=startDate, durationInYears=durationInYears, totalFunds=InitialFunds, tranchSize=TranchSize)
	fm1 = ForcastModel(tm, ForcastDuration)
	fm3 = ForcastModel(tm, ForcastDuration)
	fm5 = ForcastModel(tm, ForcastDuration)
	sequenceLabels = []
	for i in range(TradeSequenceLength): sequenceLabels += [i]
	bestActions = pd.DataFrame(columns=['date', 'neteffect','Available','Buys','Sells','Long'] + sequenceLabels)
	bestActions.set_index(['date'], inplace=True)
	if not tm.modelReady:
		print('Unable to initialize price history for model for ' + str(startDate))
		if returndailyValues: return pandas.DataFrame()
		else:return InitialFunds
	else:
		print('Calulating best actions for ' + ticker + ' from ' + str(startDate) + ' for ' + str(durationInYears) + ' years') 
		while not tm.ModelCompleted():
			day = tm.currentDate
			p = tm.GetPriceSnapshot()	#Includes today's close, so this is end of day.
			cash, asset = tm.Value()
			a,b,s,l = tm.PositionSummary()
			print(str(day)[:10],'(avail, buys, sells, long)', a,b,s,l, '(cash, asset)', cash, asset, ticker)
			bestActions.at[day,['Available','Buys','Sells','Long']] = a,b,s,l
			print(' Price Targets: ', p.nextDayTarget * (1-p.fiveDayDeviation/2), p.nextDayTarget * (1-p.fiveDayDeviation/4), p.nextDayTarget, p.nextDayTarget * (1+p.fiveDayDeviation/4), p.nextDayTarget * (1+p.fiveDayDeviation/2))
			funds = tm.FundsAvailable()
			print(' Forcasting ...') #Test for  every permutation of Buy/Sell/Hold/Cancel, get best result, then test more aggressive actions
			if a == 0:
				pooledResults = tsPool.starmap(BestSequence, [(fm3, tradeSequences3, True),(fm5, tradeSequences5, True)])
			elif a == Tranches:
				pooledResults = tsPool.starmap(BestSequence, [(fm1, tradeSequences1, True),(fm3, tradeSequences3, True)])
			else:
				pooledResults = tsPool.starmap(BestSequence, [(fm1, tradeSequences1, True),(fm3, tradeSequences3, True),(fm5, tradeSequences5, True)])
			#print(pooledResults)
			i = pooledResults.index(max(pooledResults)) #Get best result 
			bestForcastResult = pooledResults[i][0]
			bestForcastSequence = pooledResults[i][1]
			bestForcastResult, bestForcastSequence = BetterSequence(fm1, bestForcastResult, bestForcastSequence)
			print(' The best possible action is ', bestForcastSequence, ' resulting in ', bestForcastResult, ' gain')
			bestActions.at[day,'neteffect'] = bestForcastResult
			bestActions.at[day, sequenceLabels] = bestForcastSequence
			#print(bestActions)
			DoAction(tm, p, bestForcastSequence[0])	#Take first action in sequence
			tm.ProcessDay()
			print('\n')
		c, a = tm.Value()
		print('Ending Value: ', c+a, '(Cash', c, ', Asset', a, ')')
		filePath = BestActionFileName(ticker, startDate)
		bestActions.to_csv(filePath)
		if returndailyValues:
			tm.CloseModel(plotResults, saveHistoryToFile)
			return tm.GetDailyValue()   #return daily value
		else:
			return tm.CloseModel(plotResults, saveHistoryToFile)	

#------------------------------------ Supervised training -----------------------------------------------------

def TraderTrain(ticker:str, startDate:datetime, durationInYears:int, stateType:int=0, epochs:int=300, runTest:bool=True, train_test_split:float=.60, useGenericModel:bool=True, deleteExistingModel:bool=False):
	from _classes.SeriesPrediction import TradePredictionNN
	#Train model on previously generated best action sequence .csv file, output two values per day, the best action and the model predicted best action, a predictions csv file.
	#The best actions training data is for a sequence actions, but we are recording the best single action per day and the best single predicted action per day.  A trained model SHOULD predict actions for the same sequence as the training sequence.
	#The output will include both the training and test data, so noting where that cutoff occurs is significant.  The model has been told the correct answer for the training data, not so with test 
	prices = PricingData(ticker)
	prices.LoadHistory()
	prices.CalculateStats()
	prices.NormalizePrices()
	startDate = ToDate(startDate)
	endDate = AddDays(startDate, 365*durationInYears)
	if prices.historyStartDate > startDate: startDate = prices.historyStartDate
	if prices.historyEndDate < endDate: endDate = prices.historyEndDate
	filePath = BestActionFileName(ticker)
	print(filePath)
	if not FileExists(filePath): 
		CalculateBestActions(ticker, startDate, durationInYears)
		#SweepBestActionFiles(ticker)
	print('Loading action sequence from', filePath)
	targetValues = pd.read_csv(filePath, index_col=0, parse_dates=True, na_values=['nan'])
	if targetValues.index.min() > startDate: startDate = targetValues.index.min()
	if targetValues.index.max() < endDate: endDate = targetValues.index.max()
	targetValues = targetValues[startDate:endDate]
	targetValues = targetValues.rename(columns={'0':'actionID'})
	prices.TrimToDateRange(startDate, endDate)
	print('Initializing trainer for ', ticker, startDate, endDate)
	if endDate < startDate:
		print('No data found.', ticker)
		assert(False)
	stateLables = []
	for i in range(len(MakeState(stateType, None, 0))):
		stateLables += [i]
	inputStates = pd.DataFrame(columns=['date'] + stateLables)
	inputStates.set_index(['date'], inplace=True)
	dayCounter = 0
	dateIndex = prices.GetDateFromIndex(dayCounter)
	print('Populating states...') 
	a, b, s, l = 0,0,0,0
	while dayCounter < len(targetValues):	#Populate input state data.  In addition to price, state data can include available shares, pending orders, and long positions.  All part of best actions file.
		dateIndex = prices.GetDateFromIndex(dayCounter)
		p = prices.GetPriceSnapshot(forDate=dateIndex) 
		if dateIndex in targetValues.index:	
			a, b, s, l = targetValues.loc[dateIndex,['Available', 'Buys', 'Sells', 'Long']]
			state = MakeState(stateType, p, a, b, s, l)
			inputStates.at[dateIndex, stateLables] = state 
		else:
			print(' ', str(dateIndex)[:10], ticker, ' No best action data available for this date')
			targetValues.loc[dateIndex] = np.nan	#Insert an empty row to make the indexes match if necessary, will fill later
		dayCounter +=1
	targetValues = targetValues[['actionID']] #Use first action of sequence
	targetValues.fillna(method='ffill', inplace=True)
	targetValues.fillna(method='bfill', inplace=True)
	print('Prepping data ...') 
	modelName = ModelName(ticker) + '_State' + str(stateType) + '_Train'
	if useGenericModel: modelName = ModelName('Trading') + '_State' + str(stateType) + '_Train'
	model = TradePredictionNN(baseModelName=modelName, UseLSTM=UseLSTM, PredictionResultsDataFolder='data/trademodel/')
	model.LoadSource(sourceDF=inputStates, window_size=WindowSize)
	model.LoadTarget(targetDF=targetValues)
	model.MakeBatches(batch_size=64, train_test_split=train_test_split)
	if deleteExistingModel: model.SavedModelDelete()
	model.BuildModel(hidden_layer_size=512, dropout=True, dropout_rate=.01, learning_rate=2e-5)
	if (not model.Load() and epochs==0): epochs=50
	print('Training ...') 
	model.Train(epochs=epochs)
	print('Saving...') 
	model.Save()
	print('Predicting...')
	model.Predict(True)
	#model.DisplayDataSample()
	modelName = ModelName(ticker) + '_State' + str(stateType) + '_Train'
	print('Saving predictions to ' + modelName + '_predictions.csv')
	model.PredictionResultsSave(modelName + '_predictions.csv', True)
	if runTest: TraderTest(ticker=ticker, startDate=startDate.strftime('%m/%d/%Y'), durationInYears=durationInYears, testStartDate=model.test_start_date, stateType=stateType, plotResults=False, verbose=False)

#------------------------------------ Trading ------------------------------------------------------------------------
def TraderTest(ticker: str, startDate:datetime, durationInYears:int, testStartDate:datetime, stateType:int=0, plotResults:bool=False, justEvaluateInputTrainingData:bool=False, verbose:bool=False):
	#Evaluate the performance of the model using previously generated predictions.csv to specify what action to perform each day.
	if justEvaluateInputTrainingData: #Run an evaluation of the input training data to see how it performs outside the model, should be a rocket ship
		modelName = ModelName(ticker) + '_State' + str(stateType) + '_InputDataTest'
		filePath =  BestActionFileName(ticker)
		SweepBestActionFiles(ticker)
		if not FileExists(filePath): 
			CalculateBestActions(ticker, startDate, durationInYears)
			SweepBestActionFiles(ticker)
		actions = pd.read_csv(filePath, index_col=0, parse_dates=True, na_values=['nan'])
		actions['actionID_Predicted'] = actions['0']
	else:	#Evaluate the perfomance of the model itself
		modelName = ModelName(ticker) + '_State' + str(stateType) 
		filePath = 'data/trademodel/' + modelName + '_Train_predictions.csv'
		actions = pd.read_csv(filePath, index_col=0, parse_dates=True, na_values=['nan'])
		actions.fillna(value=3, inplace=True)
	startDate = ToDate(startDate)
	endDate = AddDays(startDate, days=365*durationInYears)
	if actions.index.min() > startDate: startDate = actions.index.min()
	if actions.index.max() < endDate: endDate = actions.index.max()
	tm = TradingModel(modelName=modelName + '_Test', startingTicker=ticker, startDate=startDate, durationInYears=durationInYears, totalFunds=InitialFunds, tranchSize=TranchSize, verbose=verbose)
	if not tm.modelReady:
		print('Unable to initialize price history for model for ' + str(startDate))
		return 0
	else:
		while not tm.ModelCompleted():
			day = tm.currentDate
			actionID = -1
			p = tm.GetPriceSnapshot()	
			if day < endDate and day in actions.index: actionID = int(actions['actionID_Predicted'].at[day])
			a,b,s,l = tm.PositionSummary()
			if verbose: print(a,b,s,l,'(avail, buys, sells, long) ', actionID, 'Action', day)
			DoAction(tm, p, actionID, True)
			tm.ProcessDay()			
		cash, asset = tm.Value()
		print('Ending Value: ', cash + asset, '(Cash', cash, ', Asset', asset, ')')
		tsv = tm.GetValueAt(testStartDate)
		tradeCount = len(tm.tradeHistory)
		RecordPerformance(ModelName=modelName, StartDate=startDate, TestStartDate=testStartDate, EndDate=endDate, StartValue=InitialFunds, TestStartValue=tsv, EndValue=(cash + asset), TradeCount=tradeCount)
		return tm.CloseModel(plotResults=plotResults, saveHistoryToFile=True)	
		
def TraderRun(ticker: str, startDate:datetime, durationInYears:int, stateType:int=0, useGenericModel:bool=True, calculateBestActions:bool=False, saveTraining:bool=False, plotResults:bool=False):
	#Restore model from backup, run predictions, output the result to performance.csv.  You can use a ticker specific model or generic model trained on multiple tickers
	#calculateBestActions=True will continue forecast and train the model as it moves forward in time
	#saveTraining=False specifies that ongoing training will not save changes to the model
	training_epochs=10
	from _classes.SeriesPrediction import TradePredictionNN
	modelName = ModelName(ticker) + '_State' + str(stateType)  + '_Run' + startDate[-4:]
	savedModelName = ModelName(ticker) + '_State' + str(stateType) + '_Train'
	if useGenericModel: savedModelName = ModelName('Trading') + '_State' + str(stateType) + '_Train'
	tm = TradingModel(modelName=modelName, startingTicker=ticker, startDate=startDate, durationInYears=durationInYears, totalFunds=InitialFunds, tranchSize=TranchSize, verbose=True)
	if calculateBestActions:
		ThreadCount = 3
		x = [1,3,5,7]
		tradeSequences1 = list(itertools.product([1],x,x,x))
		tradeSequences3 = list(itertools.product([3],x,x,x))
		tradeSequences5 = list(itertools.product([5],x,x,x))
		tsPool = Pool(ThreadCount)
		pooledResults = []
		fm1 = ForcastModel(tm, ForcastDuration)
		fm3 = ForcastModel(tm, ForcastDuration)
		fm5 = ForcastModel(tm, ForcastDuration)
		sequenceLabels = []
		for i in range(TradeSequenceLength): sequenceLabels += [i]
		bestActions = pd.DataFrame(columns=['date', 'neteffect','Available','Buys','Sells','Long'] + sequenceLabels)
		bestActions.set_index(['date'], inplace=True)
	stateLables = []
	for i in range(len(MakeState(stateType, None, 0))):
		stateLables += [i]
	inputStates = pd.DataFrame(columns=['date'] + stateLables)
	inputStates.set_index(['date'], inplace=True)
	targetValues = pd.DataFrame(columns=['date','actionID'])
	targetValues.set_index(['date'], inplace=True)
	rewardHistory = []
	normalizedPrices = PricingData(ticker)
	normalizedPrices.LoadHistory()
	normalizedPrices.CalculateStats()
	normalizedPrices.NormalizePrices()
	window = [] #window of WindowSize states to pass to predict
	if not tm.modelReady:
		print('Unable to initialize price history for model for ' + str(startDate))
		return 0
	else:
		print('Restoring model ...') 
		model = TradePredictionNN(baseModelName=savedModelName, UseLSTM=False, PredictionResultsDataFolder='data/trademodel/')
		model.SetModelParams(feature_count=len(MakeState(stateType, None, 0)), number_of_classes=7, window_size=WindowSize, prediction_target_days=1)
		if model.Load():
			trainCounter=-WindowSize
			while not tm.ModelCompleted():
				day = tm.currentDate
				a,b,s,l = tm.PositionSummary()
				cash, asset = tm.Value()
				print(str(day)[:10],'(avail, buys, sells, long)', a,b,s,l, '(cash, asset)', cash, asset, ticker)
				p = normalizedPrices.GetPriceSnapshot(day)	#Predict on normalizedPrices
				state = MakeState(stateType,p,a,b,s,l)
				inputStates.at[day, stateLables] = state 
				window.insert(0,state)
				if len(window) > WindowSize: del window[-1]
				if calculateBestActions:	#Use this option to train while running
					print(' Forcasting best action ...') 
					if a == 0:
						pooledResults = tsPool.starmap(BestSequence, [(fm3, tradeSequences3, False),(fm5, tradeSequences5, False)])
					elif a==Tranches:
						pooledResults = tsPool.starmap(BestSequence, [(fm1, tradeSequences1, False),(fm3, tradeSequences3, False)])
					else:
						pooledResults = tsPool.starmap(BestSequence, [(fm1, tradeSequences1, False),(fm3, tradeSequences3, False),(fm5, tradeSequences5, False)])
					i = pooledResults.index(max(pooledResults)) #Get best result 
					bestForcastResult = pooledResults[i][0]
					bestForcastSequence = pooledResults[i][1]
					bestForcastResult, bestForcastSequence = BetterSequence(fm1, bestForcastResult, bestForcastSequence)
					bestActions.at[day,['Available','Buys','Sells','Long']] = a,b,s,l
					bestActions.at[day,'neteffect'] = bestForcastResult
					bestActions.at[day, sequenceLabels] = bestForcastSequence
					bestAction = bestForcastSequence[0]
					targetValues.at[day, 'actionID'] = bestAction #Record the best action for training.  We are not going to use it.
				if len(window) == WindowSize:	#For the first WindowSize days we aren't going to do anything while we build a history window
					actionID = model.PredictOne(window)
					if calculateBestActions: 
						training_epochs += abs(bestAction-actionID)*2 #Add epochs for wrong answers
						print(' The best possible action sequence is ', bestForcastSequence, ' resulting in ', bestForcastResult, ' gain. We chose ' + str(actionID))
					else:
						print(' Choosing action ' + str(actionID))
				else:
					print(' No actions while building history window... day ', len(window))
					actionID = 3
				p = tm.GetPriceSnapshot()
				DoAction(tm, p, actionID)			
				tm.ProcessDay()
				if calculateBestActions and trainCounter > 30: #Periodically train on the data we have been getting
					print('Pausing to update training')
					targetValues = targetValues.astype({'actionID': int})
					model.LoadSource(sourceDF=inputStates, window_size=WindowSize)
					model.LoadTarget(targetDF=targetValues)
					model.MakeBatches(batch_size=32, train_test_split=1)
					training_epochs = min(training_epochs, int(len(targetValues)/10))
					model.Train(epochs=training_epochs)
					training_epochs = 10
					if saveTraining: model.Save() #To preserve integrity of prior model training don't save
					trainCounter = 0
				trainCounter +=1		
			cash, asset = tm.Value()
			print('Ending Value: ', cash + asset, '(Cash', cash, ', Asset', asset, ')')
			testStartDate = startDate	#Training was done as it came in, so as far as we know all data was test, unless it was done separately
			tsv = tm.GetValueAt(testStartDate)
			tradeCount = len(tm.tradeHistory)
			RecordPerformance(ModelName=modelName, StartDate=startDate, TestStartDate=testStartDate, EndDate=tm.currentDate, StartValue=InitialFunds, TestStartValue=tsv, EndValue=(cash + asset), TradeCount=tradeCount)
			filePath = BestActionFileName(ticker, startDate) + '_Run'	#These are only the best actions from current portfolio which may not have been well managed
			if calculateBestActions: bestActions.to_csv(filePath)
			return tm.CloseModel(plotResults=plotResults, saveHistoryToFile=True)	

def TraderRandom(ticker: str, startDate:datetime, testStartDate:datetime, durationInYears:int, plotResults:bool=False, verbose:bool=True):
	#Pick random actions for each day to see this performs
	modelName = 'Random_' + (ticker) + '_' + startDate[-4:]
	tm = TradingModel(modelName, ticker, startDate, durationInYears, InitialFunds, TranchSize)
	if not tm.modelReady:
		print('Unable to initialize price history for model for ' + str(startDate))
		return 0
	else:
		while not tm.ModelCompleted():
			day = tm.currentDate
			c, a = tm.Value()
			a,b,s,l = tm.PositionSummary()
			p = tm.GetPriceSnapshot()	
			actionID = randint(0, 6)
			if verbose and not actionID == 3: print(a,b,s,l,'(avail, buys, sells, long) ', actionID, 'Action: ', day)
			DoAction(tm, p, actionID)
			tm.ProcessDay()
		cash, asset = tm.Value()
		print('Ending Value: ', cash + asset, '(Cash', cash, ', Asset', asset, ')')
		tsv = tm.GetValueAt(testStartDate)
		tradeCount = len(tm.tradeHistory)
		RecordPerformance(ModelName=modelName, StartDate=startDate, TestStartDate=testStartDate, EndDate=tm.currentDate, StartValue=InitialFunds, TestStartValue=tsv, EndValue=(cash + asset), TradeCount=tradeCount)
		return tm.CloseModel(plotResults=plotResults, saveHistoryToFile=True)	

def TraderBuyHold(ticker: str, startDate:str, testStartDate:datetime, durationInYears:int, plotResults:bool=False, verbose:bool=False):
	#Baseline model for comparison, it is pointless to trade if you can't beat this
	modelName = 'BuyHold_' + (ticker) + '_' + startDate[-4:]
	tm = TradingModel(modelName, ticker, startDate, durationInYears, InitialFunds, TranchSize)
	if not tm.modelReady:
		print('Unable to initialize price history for model for ' + str(startDate))
		return 0
	else:
		i = 0
		while not tm.ModelCompleted():
			day = tm.currentDate
			if i <= Tranches:
				p = tm.GetPriceSnapshot()	
				DoAction(tm, p, 2)
				i +=1 
			tm.ProcessDay()
		cash, asset = tm.Value()
		print('Ending Value: ', cash + asset, '(Cash', cash, ', Asset', asset, ')')
		tsv = tm.GetValueAt(testStartDate)
		tradeCount = len(tm.tradeHistory)
		RecordPerformance(ModelName=modelName, StartDate=startDate, TestStartDate=testStartDate, EndDate=tm.currentDate, StartValue=InitialFunds, TestStartValue=tsv, EndValue=(cash + asset), TradeCount=tradeCount)
		return tm.CloseModel(plotResults=plotResults, saveHistoryToFile=True)	

def ExtensivePreparation(tickerList:list):
	#Prepare the best action files for training, if they don't already exist.  This takes time as it has to test every permutation to see which produces the best result.
	#It will then train the model using the best action files as input
	for ticker in tickerList:
		for i in range(5):
			startDate='1/1/' + str(1990 + i*5)
			filePath = BestActionFileName(ticker, startDate)
			if not os.path.isfile(filePath): CalculateBestActions(ticker=ticker, startDate=startDate, durationInYears=5)
		filePath = BestActionFileName(ticker)
		if not os.path.isfile(filePath): SweepBestActionFiles(ticker)
		if os.path.isfile(filePath): 
			TraderTrain(ticker=ticker, startDate='1/1/1990', durationInYears=20, stateType=6, epochs=350, train_test_split=.60, useGenericModel=False)

def ExtensiveTesting(tickerList:list):
	#This will run the model through various date ranges, compare it to BuyHold and Random, and record the results to data/trademodel/performance.csv 
	stateType=1
	for ticker in tickerList:
		filePath = BestActionFileName(ticker)
		if os.path.isfile(filePath):
			startDate = '1/1/2020'
			TraderRun(ticker=ticker, startDate=startDate, durationInYears=8, stateType=stateType, useGenericModel=True, calculateBestActions=False, saveTraining=False, plotResults=False)
			TraderBuyHold(ticker=ticker, startDate=startDate, testStartDate=startDate, durationInYears=8)
			TraderRandom(ticker=ticker, startDate=startDate, testStartDate=startDate, durationInYears=8)
			for i in range(2):
				startDate='1/1/' + str(2005 + i*5)	#2005 to 2015
				TraderRun(ticker=ticker, startDate=startDate, durationInYears=5, stateType=stateType, useGenericModel=True, calculateBestActions=False, saveTraining=False, plotResults=False)
				TraderBuyHold(ticker=ticker, startDate=startDate, testStartDate=startDate, durationInYears=5)
				TraderRandom(ticker=ticker, startDate=startDate, testStartDate=startDate, durationInYears=5)
			for i in range(3):
				startDate='1/1/' + str(1990 + i*5)	#1990 to 2000
				TraderRun(ticker=ticker, startDate=startDate, durationInYears=5, stateType=stateType, useGenericModel=True, calculateBestActions=False, saveTraining=False, plotResults=False)
				TraderBuyHold(ticker=ticker, startDate=startDate, testStartDate=startDate, durationInYears=5)
				TraderRandom(ticker=ticker, startDate=startDate, testStartDate=startDate, durationInYears=5)
				
def BaselineTests(tickerList:list):
	for ticker in tickerList:
		filePath = BestActionFileName(ticker)
		if os.path.isfile(filePath):
			durationInYears = 25
			startDate = '3/27/1990'
			endDate = AddDays(startDate, durationInYears*365)
			testStartDate = TestStartDate(startDate, endDate)
			TraderBuyHold(ticker=ticker, startDate=startDate, testStartDate=testStartDate, durationInYears=durationInYears)
			TraderRandom(ticker=ticker,  startDate=startDate, testStartDate=testStartDate, durationInYears=durationInYears)

if __name__ == '__main__':
	multiprocessing.freeze_support()
	#tickerList=['BAC','XOM','JNJ']
	tickerList=['JNJ']
	ExtensivePreparation(tickerList)
	ExtensiveTesting(tickerList)
	BaselineTests(tickerList)