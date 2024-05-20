# Stock Price Prediction Project
The main goal of this project is to explore methods to augment stock price data with other data sets to determine whether augmenting data sets will improve prediction accuracy. For this project, historical stock price data will be combined with analysts’ buy/sell/hold ratings for that stock at that price. The augmented data will then be inputted into a Long-Short Term Memory (LSTM) model for training and prediction. 
## User Guide
1. Ensure python3 is installed
2. Ensure all the .csv files are in the same directory as the main.py file
3. Run the following: ./python3 main.py
4. When prompted “Enter stock ticker: ”, type in the stock ticker to run the predictor on. There are three options to choose from:
  * AAPL
  * INTC
  * NKE
5. The program will generate two graphs. The first graph will be for the LSTM model
without analyst rating data and the mean squared error and mean absolute
percentage error are printed in the console. Once you are done viewing, exit out of
the first graph, which should then resume the prediction program to run on the
LSTM model with analyst rating data. The second graph will be the graph for the
LSTM model with analyst rating data with the mean squared error and mean
absolute percentage error are printed in the console as well.
