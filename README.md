# MEXC Advanced Trading Bot

Bot trading otomatis untuk MEXC Exchange dengan menggunakan analisis teknikal lanjutan dan model machine learning.

## Fitur Utama

### 📊 75+ Indikator Teknikal Canggih
- **Momentum**: RSI, Stochastic, Williams %R, TSI, Ultimate Oscillator
- **Trend**: MACD, Moving Averages, ADX, Ichimoku Cloud, Parabolic SAR, KST, DPO
- **Volatilitas**: Bollinger Bands, ATR, Keltner Channel, Donchian Channel
- **Volume**: OBV, MFI, VWAP, Chaikin Money Flow, Force Index
- **Deteksi Pola Candlestick**: Doji, Hammer, Engulfing, Morning Star

### 🧠 Advanced Feature Engineering
- Deteksi market regime (trending/ranging/volatile)
- Interaction features (kombinasi indikator)
- Fitur multi-timeframe
- Lagged features untuk analisis time series

### 🏆 Kombinasi Model Machine Learning
- **Ensemble Model**: Random Forest, Gradient Boosting, XGBoost, SVM
- **Deep Learning**: LSTM neural network untuk analisis time series
- **Algoritma Voting Terboboti** untuk kombinasi prediksi

### 📈 Analisis Backtest Komprehensif
- Equity curve dengan perbandingan HODL
- Analisis distribusi profit/loss
- Analisis berdasarkan alasan exit (TP/SL/Signal)
- Perhitungan fee trading dan metrics penting

## Instalasi

1. Clone repository:
```bash
git clone https://github.com/username/MEXCTrading.git
cd MEXCTrading
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Konfigurasi

Edit parameter di awal file `main.py`:

```python
# --- Trading Pairs & Timeframe ---
PAIR = 'MX/USDT'           # Pair trading
TIMEFRAME = '15m'          # Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
DATA_LIMIT = 1000          # Jumlah candle untuk analisis

# --- API Keys ---
API_KEY = 'YOUR_API_KEY'    # API Key MEXC
API_SECRET = 'YOUR_SECRET'  # API Secret MEXC

# --- Risk Management ---
TRADE_RISK = 0.01           # Max 1% dari modal per posisi
STOP_LOSS_PCT = 0.005       # 0.5% stop-loss
TAKE_PROFIT_PCT = 0.012     # 1.2% take-profit
TRADE_AMOUNT = 12           # $12/trade
```

## Penggunaan

### Mode Simulasi (Backtest)
```bash
python main.py
```

### Mode Live Trading
```bash
python main.py --live
```

## Hasil Backtest

Bot telah diuji dengan berbagai pair dan timeframe, dengan hasil terbaik:
- BTC/USDT 4h: Profit +5.17% dengan win rate 77.04%
- Perbandingan dengan HODL menunjukkan bot bekerja baik dalam sideways/bearish market

## Struktur Direktori

- `main.py`: Kode utama bot
- `results/`: Direktori untuk menyimpan hasil backtest, model ML, dan laporan
- `requirements.txt`: Dependensi yang diperlukan

## Perhatian dan Disclaimer

Bot trading ini masih dalam tahap pengembangan. Gunakan dengan risiko Anda sendiri:
- Mulai dengan jumlah kecil untuk testing
- Monitor performa bot secara berkala
- Perhatikan kondisi market secara umum

Bot bekerja paling baik dalam kondisi sideways market, dan mungkin underperform dibanding HODL saat bull market ekstrim.

## License

MIT 