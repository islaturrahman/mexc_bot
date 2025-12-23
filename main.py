import os
import time 
import pytz
import  datetime
import json
import platforms.mexcspot.mexc_spot as mexc
import platforms.mexcspot.config as config


from backtest import BacktestMexc
from platforms.mexcspot.mexc_strategy import (SimpleRSIStrategy,
                           RSIMeanReversionStrategy,
                           AlwaysInMarketStrategy,
                           RSITrendFollowingStrategy,
                           SupertrendFootprintStrategy,
                           BollingerMacdStrategy,
                            TurtleTrading1,
                            TurtleTradingOptimized,
                            TurtleClassic
                           )

from platforms.mexcspot.ml_strategy import CPOStrategy

from platforms.mexcspot.mexc_live_trade import MexcTurtleBot

class MexcRobot():
    def __init__(self, symbol, interval): 
        self.api_key = config.api_key
        self.secret_key = config.secret_key
        self.mexc_host = config.mexc_host

        self.symbol = symbol
        self.interval = interval

    def mexc_backtesting_strategy(self):
        backtester = BacktestMexc(
            symbol="BNBUSDT",
            interval="5m",
            limit=1000,
            cash=1000000,
            days_ago=10,
            commission=0.0005,
            strategy=TurtleTradingOptimized
        )
        filename = 'data/ohlcv_data.csv'

        print("="*60)
        print("MEXC DATA & BACKTESTING SYSTEM")
        print("="*60)

        if not os.path.exists(filename):
            print(f"[INFO] File {filename} tidak ditemukan. Mendownload data dari MEXC...")
            backtester.save_ohlcv_to_csv(filename=filename)
        else:
            print(f"[OK] Menggunakan file {filename} yang sudah ada.")

        
        print("\n[STEP 1] Checking RSI signals in data...")
        backtester.check_rsi_signals(filename)
        

        print("\n[STEP 2] Running Backtest ...{back}")
        stats, bt = backtester.run_backtest()
        
        print("\n" + "="*60)
        print("PROSES SELESAI")
        print("="*60)


    def mexc_live_trade(self):
        live_trader = MexcTurtleBot(
            symbol=self.symbol,
            interval=self.interval
        )
        live_trader.run()

if __name__ == "__main__":
    mexc = MexcRobot(
        symbol="BNBUSDT",   
        interval="1m"
    )
    mexc.mexc_live_trade()

    # mexc.mexc_backtesting_strategy()





    