import yfinance as yf

# 获得雅虎数据
ticker = yf.Ticker("601318.SS")
data = ticker.history(period="max")

print(data.head())
