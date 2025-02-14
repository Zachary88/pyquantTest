import yfinance as yf

# 获得雅虎数据
ticker = yf.Ticker("M2505.DCE")
data = ticker.history(period="1mo", interval="1d")

print(data[["Open", "High", "Low", "Close", "Volume"]])

