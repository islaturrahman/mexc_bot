import mexc_spot as mexc
import mexc_strategy as strategy

from backtest import BacktestMexc
from mexc_strategy import SimpleRSIStrategy,RSIMeanReversionStrategy,AlwaysInMarketStrategy,RSITrendFollowingStrategy,SupertrendFootprintStrategy

import os

class MexcRobot:
    def __init__(self):
        self.api_key = mexc.config.api_key
        self.secret_key = mexc.config.secret_key
        self.mexc_host = mexc.config.mexc_host

        self.market = mexc.mexc_market()
        self.account = mexc.mexc_trade()

    def get_ping(self):
        return self.market.get_ping()
    
    def fetch_ticker(self, symbol):
        params = {
            "symbol": symbol,
            "limit": 500,
            "interval": "1m"
        }
        return self.market.get_kline(params=params)

    def get_account_info(self):
        return self.account.get_account_info()

    def get_orders(self, symbol):
        params = {
            "symbol": symbol,
            "limit": 500,
            "page": 1
        }
        return self.account.get_order(params=params)


    def get_fee(self, symbol):
        params = {
            "symbol": symbol,
            "timestamp": self.market.get_timestamp()
        }
        return self.account.get_symbol_commission(params=params)



if __name__ == "__main__":
    # robot = MexcRobot()
    
    backtester = BacktestMexc(
        symbol="BTCUSDT",
        interval="30m",
        limit=1000,
        cash=1000000,
        days_ago=300,
        commission=0.001,
        strategy=SupertrendFootprintStrategy
    )
    
    filename = 'data/ohlcv_data.csv'

    print("="*60)
    print("MEXC DATA & BACKTESTING SYSTEM")
    print("="*60)
    
    # === STEP 0: DOWNLOAD DATA JIKA BELUM ADA ===
    if not os.path.exists(filename):
        print(f"[INFO] File {filename} tidak ditemukan. Mendownload data dari MEXC...")
        # Ambil data 350 hari ke belakang (sesuai setting kamu)
        backtester.save_ohlcv_to_csv(filename=filename)
    else:
        print(f"[OK] Menggunakan file {filename} yang sudah ada.")

    # === STEP 1: ANALISIS SINYAL ===
    print("\n[STEP 1] Checking RSI signals in data...")
    backtester.check_rsi_signals(filename)
    
    # === STEP 2: RUN BACKTEST DENGAN SIZE FIX ===
    # Gunakan modal $89 sesuai saldo akun MEXC kamu
    print("\n[STEP 2] Running Backtest (RSI Strategy)...")
    stats, bt = backtester.run_backtest()
    
    print("\n" + "="*60)
    print("PROSES SELESAI")
    print("="*60)