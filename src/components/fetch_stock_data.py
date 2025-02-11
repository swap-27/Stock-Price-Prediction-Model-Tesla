import yfinance as yf
import datetime
from sklearn.model_selection import train_test_split

class StockPriceDataFetch:
    def __init__(self):



        self.end_date_main = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
        #self.end_date_main = datetime.datetime.strptime(self.end_date, '%Y-%m-%d') - datetime.timedelta(days = 1)

        #getting data
        self.df = yf.download('TSLA', start='2010-06-29', end=self.end_date_main)


        self.df.columns = self.df.columns.droplevel(1)
        self.close_df = self.df['Close']