import ccxt
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import joblib
import sys
import xgboost as xgb

# Cek ketersediaan GPU untuk XGBoost
try:
    gpu_available = xgb.config_context(verbosity=0)
    # Test if GPU available
    test_param = {'tree_method': 'gpu_hist', 'gpu_id': 0}
    xgb.train({'tree_method': 'gpu_hist'}, xgb.DMatrix(np.random.randn(10, 10), np.random.randn(10)))
    print("XGBoost GPU support tersedia dan aktif")
    XGB_GPU_AVAILABLE = True
except Exception as e:
    print(f"XGBoost GPU tidak tersedia (akan menggunakan CPU): {e}")
    XGB_GPU_AVAILABLE = False

# Coba impor TensorFlow/Keras dengan penanganan kesalahan
try:
    # type: ignore # Menandai untuk IDE bahwa ini OK walaupun tidak bisa diresolve
    import tensorflow as tf
    # Cek ketersediaan GPU untuk TensorFlow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Izinkan memory growth untuk GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"TensorFlow GPU tersedia: {len(gpus)} GPU ditemukan")
            TF_GPU_AVAILABLE = True
        except RuntimeError as e:
            print(f"Error konfigurasi GPU TensorFlow: {e}")
            TF_GPU_AVAILABLE = False
    else:
        print("TensorFlow akan menggunakan CPU")
        TF_GPU_AVAILABLE = False
        
    # type: ignore
    from tensorflow.keras.models import Sequential
    # type: ignore
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
    # type: ignore
    from tensorflow.keras.optimizers import Adam
    # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"TensorFlow tidak tersedia, fitur deep learning tidak akan digunakan: {e}")
    TENSORFLOW_AVAILABLE = False
    TF_GPU_AVAILABLE = False

warnings.filterwarnings("ignore")

# Periksa command line arguments
if len(sys.argv) > 1 and sys.argv[1] == '--live':
    print("=== MODE LIVE TRADING DIAKTIFKAN ===")
    IS_LIVE_DEFAULT = True
else:
    IS_LIVE_DEFAULT = False

# ========== PARAMETER ==========
# --- Trading Pairs & Timeframe ---
PAIR = 'MANTA/USDT'            # <---- Pair dengan performa terbaik di timeframe 4h
TIMEFRAME = '1h'             # <---- Timeframe optimal berdasarkan analisis
DATA_LIMIT = 500

# --- API Keys ---
API_KEY = 'Jy2D5DAbVmti5ZkuiccnUKzdgmbRapji1iEgiGYWYAaPnY3UmLk0dAdDUf5msHVv'
API_SECRET = 'z2vAgYukrEsSBBhuw7Yiex3JtFVikG8sZR3fnwZQHYttXire59zQgH50ReU78m7X'

# --- Risk Management ---
MIN_BALANCE = 10            # Minimal usdt yang harus tersedia untuk entry
TRADE_RISK = 0.01            # Maks 1% dari modal per posisi
STOP_LOSS_PCT = 0.005        # 0.5% stop-loss
TAKE_PROFIT_PCT = 0.012      # 1.2% take-profit (ditingkatkan karena performa TP yang bagus di 4h)
TRADE_AMOUNT = 10            # $12/trade (disesuaikan, jangan seluruh saldo)
TRADING_FEE = 0.002          # Fee trading MEXC (0.2% per transaksi)

# --- Machine Learning Parameters ---
TARGET_THRESHOLD = 0.001     # Threshold untuk label target (0.1% pergerakan minimal)
N_ESTIMATORS = 100           # Jumlah trees di Random Forest (default)
N_ESTIMATORS_GPU = 200       # Jumlah trees di Random Forest saat GPU tersedia
MAX_FEATURES = 0.7           # Persentase fitur yang digunakan (0.7 = 70%)
TOP_FEATURES_PCT = 0.5       # Ambil 50% top features saja untuk mengurangi overfitting

# --- Notification ---
TG_TOKEN = '7859663172:AAEsL9e0tp3azEdYp6-BdeIjSDZ9pPqcgcQ'
TG_CHATID = '6271836846'

# Fungsi untuk cek koneksi Telegram
def check_telegram_connection():
    try:
        import requests
        url = f'https://api.telegram.org/bot{TG_TOKEN}/getMe'
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['ok']:
                bot_name = data['result']['first_name']
                bot_username = data['result']['username']
                print(f"[Telegram] Koneksi berhasil! Bot name: {bot_name} (@{bot_username})")
                # Kirim pesan tes
                send_telegram(f"🔄 Bot Trading MEXC terhubung!\nBot siap menerima notifikasi.")
                return True
            else:
                print(f"[Telegram] Koneksi gagal: {data.get('description', 'Unknown error')}")
        else:
            print(f"[Telegram] Koneksi gagal. Status code: {response.status_code}")
        return False
    except Exception as e:
        print(f"[Telegram] Cek koneksi gagal: {str(e)}")
        return False

# --- Mode Run ---
IS_LIVE = IS_LIVE_DEFAULT  # Akan diaktifkan dengan parameter --live
SAVE_MODEL = True            # Simpan model ML untuk digunakan lagi
PLOT_RESULTS = True          # Plot equity curve dan metrik lainnya
SAVE_RESULTS = True          # Simpan hasil backtest ke CSV

# --- Aktifkan bot dengan perintah: ---
# Simulasi: python main.py
# Live Trading: python main.py --live

# --- Output Directory ---
OUTPUT_DIR = 'results'       # Direktori untuk menyimpan hasil
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# ===============================

def send_telegram(msg):
    try:
        import requests
        url = f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage'
        payload = {'chat_id': TG_CHATID, 'text': msg, 'parse_mode':'HTML'}
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code == 200:
            print(f"[Telegram] Notifikasi berhasil dikirim")
        else:
            print(f"[Telegram] Gagal mengirim notifikasi. Status code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print("[Telegram] Gagal kirim notifikasi:",str(e))

def get_data_ccxt(symbol=PAIR, timeframe=TIMEFRAME, limit=DATA_LIMIT):
    ex = ccxt.binance({'enableRateLimit': True})
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def feature_engineering(df, multi_timeframe_data=None):
    """
    Feature engineering canggih dengan 75+ indikator teknikal
    """
    # Clone dataframe untuk menghindari warning
    df = df.copy()
    
    try:
        # ===== 1. INDIKATOR MOMENTUM =====
        # RSI dengan berbagai periode
        if len(df) >= 21:
            df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
            df['rsi_7'] = ta.momentum.rsi(df['close'], window=7)
            df['rsi_21'] = ta.momentum.rsi(df['close'], window=21)
        else:
            df['rsi_14'] = np.nan
            df['rsi_7'] = np.nan
            df['rsi_21'] = np.nan
        # Stochastic Oscillator
        if len(df) >= 14:
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
        else:
            df['stoch_k'] = np.nan
            df['stoch_d'] = np.nan
        # Williams %R
        if len(df) >= 14:
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
        else:
            df['williams_r'] = np.nan
        # True Strength Index (TSI)
        if len(df) >= 25:
            df['tsi'] = ta.momentum.tsi(df['close'], window_slow=25, window_fast=13)
        else:
            df['tsi'] = np.nan
        # Rate of Change
        if len(df) >= 20:
            df['roc_10'] = ta.momentum.roc(df['close'], window=10)
            df['roc_20'] = ta.momentum.roc(df['close'], window=20)
        else:
            df['roc_10'] = np.nan
            df['roc_20'] = np.nan
        # Percentage Price Oscillator (PPO)
        if len(df) >= 26:
            df['ppo'] = ta.momentum.ppo(df['close'])
        else:
            df['ppo'] = np.nan
        # Ultimate Oscillator
        if len(df) >= 7:
            df['ultimate_osc'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'])
        else:
            df['ultimate_osc'] = np.nan
        # Awesome Oscillator
        if len(df) >= 5:
            df['awesome_osc'] = ta.momentum.awesome_oscillator(df['high'], df['low'])
        else:
            df['awesome_osc'] = np.nan
        # ===== 2. INDIKATOR TREND =====
        # Moving Averages dengan berbagai periode
        if len(df) >= 50:
            df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
            df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
            df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        else:
            df['sma_10'] = np.nan
            df['sma_20'] = np.nan
            df['sma_50'] = np.nan
            df['ema_10'] = np.nan
            df['ema_20'] = np.nan
            df['ema_50'] = np.nan
        # MACD
        if len(df) >= 26:
            macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
        else:
            df['macd'] = np.nan
            df['macd_signal'] = np.nan
            df['macd_diff'] = np.nan
        # ADX (Average Directional Index)
        if len(df) >= 14:
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
        else:
            df['adx'] = np.nan
            df['adx_pos'] = np.nan
            df['adx_neg'] = np.nan
    except Exception as e:
        print(f"Error dalam kalkulasi indikator trend/momentum: {e}")
    
    try:
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        
        # Parabolic SAR
        df['psar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
        
        # KST Oscillator
        df['kst'] = ta.trend.kst(df['close'])
        df['kst_sig'] = ta.trend.kst_sig(df['close'])
        
        # DPO (Detrended Price Oscillator)
        df['dpo'] = ta.trend.dpo(df['close'])
        
        # Vortex Indicator
        df['vortex_pos'] = ta.trend.vortex_indicator_pos(df['high'], df['low'], df['close'], window=14)
        df['vortex_neg'] = ta.trend.vortex_indicator_neg(df['high'], df['low'], df['close'], window=14)
        
        # TRIX (Triple Exponential Average)
        df['trix'] = ta.trend.trix(df['close'], window=14)
    except Exception as e:
        print(f"Error dalam kalkulasi indikator Ichimoku/SAR: {e}")
    
    try:
        # ===== 3. INDIKATOR VOLATILITAS =====
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_pct'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # Average True Range (ATR)
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr_pct'] = df['atr'] / df['close'] * 100  # ATR sebagai persentase dari harga
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=20)
        df['kc_high'] = keltner.keltner_channel_hband()
        df['kc_low'] = keltner.keltner_channel_lband()
        df['kc_width'] = (df['kc_high'] - df['kc_low']) / keltner.keltner_channel_mband()
        
        # Donchian Channel
        df['dc_high'] = ta.volatility.donchian_channel_hband(df['high'], df['low'], df['close'], window=20)
        df['dc_low'] = ta.volatility.donchian_channel_lband(df['high'], df['low'], df['close'], window=20)
        df['dc_mid'] = ta.volatility.donchian_channel_mband(df['high'], df['low'], df['close'], window=20)
    except Exception as e:
        print(f"Error dalam kalkulasi indikator volatilitas: {e}")
    
    try:
        # ===== 4. INDIKATOR VOLUME =====
        if 'volume' in df.columns:
            # On-Balance Volume (OBV)
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            
            # Money Flow Index (MFI)
            df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
            
            # Accumulation/Distribution
            df['ad'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
            
            # Chaikin Money Flow
            df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=20)
            
            # Ease of Movement
            df['eom'] = ta.volume.ease_of_movement(df['high'], df['low'], df['volume'], window=14)
            
            # Force Index
            df['fi'] = ta.volume.force_index(df['close'], df['volume'], window=13)
            
            # Volume Price Trend
            df['vpt'] = ta.volume.volume_price_trend(df['close'], df['volume'])
            
            # Negative Volume Index
            df['nvi'] = ta.volume.negative_volume_index(df['close'], df['volume'])
    except Exception as e:
        print(f"Error dalam kalkulasi indikator volume: {e}")
    
    try:
        # ===== 5. PRICE-DERIVED FEATURES =====
        # Price change features
        df['close_pct_change'] = df['close'].pct_change()
        df['open_pct_change'] = df['open'].pct_change()
        if 'volume' in df.columns:
            df['volume_pct_change'] = df['volume'].pct_change()
        
        # High-Low range
        df['high_low_diff'] = (df['high'] - df['low']) / df['low']
        df['close_open_diff'] = abs(df['close'] - df['open']) / df['open']
        
        # Price position features
        df['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        df['close_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Lagged features (untuk time series)
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'close_pct_lag_{lag}'] = df['close_pct_change'].shift(lag)
            if 'volume' in df.columns:
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Return calculations
        for period in [2, 3, 5, 10]:
            df[f'return_{period}'] = df['close'].pct_change(period)
    except Exception as e:
        print(f"Error dalam kalkulasi price-derived features: {e}")
    
    try:
        # ===== 6. CANDLESTICK PATTERNS =====
        # Deteksi pola candlestick sederhana
        # Doji: open dan close hampir sama
        df['doji'] = ((abs(df['close'] - df['open']) / (df['high'] - df['low'])) < 0.1).astype(int)
        
        # Hammer: lower shadow jauh lebih panjang dari body dan upper shadow
        df['hammer'] = (((df['low'] < df['open']) & (df['low'] < df['close'])) & 
                        ((df['high'] - df['close'] < abs(df['close'] - df['open']) * 0.3) | 
                         (df['high'] - df['open'] < abs(df['close'] - df['open']) * 0.3)) &
                        ((df['close'] > df['open']) & (df['open'] - df['low'] > abs(df['close'] - df['open']) * 2)) |
                        ((df['open'] > df['close']) & (df['close'] - df['low'] > abs(df['close'] - df['open']) * 2))
                       ).astype(int)
        
        # Bullish Engulfing
        df['engulfing'] = (((df['close'] > df['open']) & 
                            (df['open'].shift(1) > df['close'].shift(1)) &
                            (df['close'] > df['open'].shift(1)) & 
                            (df['open'] < df['close'].shift(1)))
                          ).astype(int)
        
        # Morning Star (sederhana)
        df['morning_star'] = (((df['close'].shift(2) < df['open'].shift(2)) &  # Candle pertama: bearish
                               (abs(df['close'].shift(1) - df['open'].shift(1)) < abs(df['close'].shift(2) - df['open'].shift(2)) * 0.3) &  # Candle kedua: kecil
                               (df['close'] > df['open']) &  # Candle ketiga: bullish
                               (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2))  # Candle ketiga: close di atas midpoint candle pertama
                             ).astype(int)
    except Exception as e:
        print(f"Error dalam kalkulasi candlestick patterns: {e}")
    
    try:
        # ===== 7. MARKET REGIME DETECTION =====
        # Volatility regime (high/low)
        if 'atr_pct' in df.columns:
            df['volatility_regime'] = np.where(df['atr_pct'] > df['atr_pct'].rolling(30).mean(), 1, 0)
        
        # Trend regime using ADX
        if 'adx' in df.columns:
            df['trend_regime'] = np.where(df['adx'] > 25, 1, 0)
        
        # Range-bound regime using Bollinger Width
        if 'bb_width' in df.columns:
            df['range_regime'] = np.where(df['bb_width'] < df['bb_width'].rolling(30).mean(), 1, 0)
    except Exception as e:
        print(f"Error dalam kalkulasi market regime: {e}")
    
    try:
        # ===== 8. INTERACTION FEATURES =====
        # Combining indicators for enhanced signals
        if 'rsi_14' in df.columns and 'sma_20' in df.columns:
            df['rsi_sma_cross'] = np.where(df['rsi_14'] > df['sma_20'], 1, -1)
        
        if 'macd' in df.columns and 'rsi_14' in df.columns:
            df['macd_rsi'] = df['macd'] * df['rsi_14']
        
        if 'adx' in df.columns and 'rsi_14' in df.columns:
            df['adx_rsi'] = df['adx'] * df['rsi_14']
    except Exception as e:
        print(f"Error dalam kalkulasi interaction features: {e}")
    
    try:
        # ===== 9. MULTI-TIMEFRAME FEATURES =====
        if multi_timeframe_data is not None:
            # Assuming multi_timeframe_data is a dict with keys like '1h', '4h', '1d'
            for tf, tf_df in multi_timeframe_data.items():
                # Resample indicators to match our primary timeframe
                if not tf_df.empty:
                    try:
                        # RSI
                        tf_df['rsi_14'] = ta.momentum.rsi(tf_df['close'], window=14)
                        # MACD
                        tf_macd = ta.trend.MACD(tf_df['close'])
                        tf_df['macd'] = tf_macd.macd()
                        # ADX
                        tf_adx = ta.trend.ADXIndicator(tf_df['high'], tf_df['low'], tf_df['close'])
                        tf_df['adx'] = tf_adx.adx()
                        
                        # Resample to our timeframe
                        for col in ['rsi_14', 'macd', 'adx']:
                            # Forward fill the values
                            resampled = tf_df[col].reindex(df.index, method='ffill')
                            df[f'{col}_{tf}'] = resampled
                    except Exception as e:
                        print(f"Error kalkulasi indikator untuk timeframe {tf}: {e}")
    except Exception as e:
        print(f"Error dalam multi-timeframe features: {e}")
    
    # Hapus NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Cek kolom yang seharusnya ada dalam dataframe
    essential_columns = [
        'rsi_14', 'macd', 'adx', 'bb_width', 'atr', 'obv', 'ad',
        'close_pct_change', 'high_low_diff', 'close_to_high', 'close_to_low',
        'close_pct_lag_1', 'close_pct_lag_2', 'close_pct_lag_3', 'close_pct_lag_5',
        'return_2', 'return_3', 'return_5', 'rsi_7', 'rsi_21', 
        'stoch_k', 'stoch_d', 'williams_r', 'adx_pos', 'adx_neg', 
        'macd_signal', 'macd_diff', 'bb_high', 'bb_low', 'bb_mid',
        'bb_pct', 'atr_pct', 'mfi', 'cmf', 'fi',
        'doji', 'hammer', 'engulfing', 'morning_star',
        'volatility_regime', 'trend_regime', 'range_regime'
    ]
    
    # Tambahkan kolom esensial jika tidak ada (dengan nilai 0)
    for col in essential_columns:
        if col not in df.columns:
            print(f"Menambahkan kolom esensial yang hilang: {col}")
            df[col] = 0
    
    return df.dropna()

def label_target(df, threshold=TARGET_THRESHOLD):
    """
    Label data berdasarkan pergerakan harga.
    threshold: Ambang persentase kenaikan/penurunan minimal untuk label (0.0 = naik/turun)
    """
    # Versi sederhana: 1 jika naik, 0 jika turun
    if threshold == 0.0:
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    else:
        # Versi dengan threshold: 1 jika naik > threshold%, 0 jika turun > threshold%, -1 jika sideways
        df['pct_change'] = df['close'].pct_change(1).shift(-1)
        df['target'] = np.where(df['pct_change'] > threshold, 1, 
                      np.where(df['pct_change'] < -threshold, 0, -1))
        # Filter hanya data dengan target 0 atau 1 (hapus sideways)
        df = df[df['target'] != -1]
    
    return df.dropna()

def select_features(df, feature_importance=None, top_pct=TOP_FEATURES_PCT):
    """
    Pilih fitur berdasarkan importance atau semua fitur jika tidak ada importance.
    """
    # Exclude these columns from features
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 
                    'target', 'pct_change', 'pred', 'signal', 'returns', 'strategy']
    
    # If we have feature importance, select top features directly
    if feature_importance is not None:
        # Konversi ke Series jika feature_importance adalah DataFrame
        if isinstance(feature_importance, pd.DataFrame):
            # Jika feature_importance adalah DataFrame dengan satu kolom
            if len(feature_importance.columns) == 1:
                feature_importance = feature_importance.iloc[:, 0]  # Ambil kolom pertama
            # Jika feature_importance adalah DataFrame dengan 0 sebagai kolom index
            elif 0 in feature_importance.columns:
                feature_importance = feature_importance[0]
            else:
                print("Warning: Feature importance format tidak dikenal, menggunakan kolom pertama")
                feature_importance = feature_importance.iloc[:, 0]
        
        # Sort features by importance
        sorted_features = feature_importance.sort_values(ascending=False)
        # Select top N% features
        n_features = max(3, int(len(sorted_features) * top_pct))  # At least 3 features
        selected_features = sorted_features.index[:n_features].tolist()
        print(f"Menggunakan {n_features} fitur teratas dari total {len(sorted_features)} fitur")
        return selected_features
    
    # Jika tidak ada feature_importance, gunakan semua fitur dari DataFrame
    if df is not None:
        # Get all potential features
        all_features = [col for col in df.columns if col not in exclude_cols]
        return all_features
    
    # Jika tidak ada feature_importance dan df juga None
    print("Warning: Tidak ada feature importance atau dataframe yang valid")
    return []

def train_ml_advanced(df, use_cv=True, feature_importance=None):
    """
    Train advanced ML models dengan ensemble dan deep learning
    """
    # Pilih fitur (semua indikator teknikal atau dari feature importance)
    features = select_features(df, feature_importance)
    
    X = df[features]
    y = df['target']
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)
    
    # Split data for validation (untuk final evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, shuffle=False)
    
    # Dictionary untuk model ensemble
    models = {}
    predictions = {}
    
    # 1. Train Random Forest
    print("Training Random Forest...")
    # RF tidak mendukung GPU secara native, tetapi kita bisa meningkatkan kompleksitas jika ada GPU
    n_estimators_to_use = N_ESTIMATORS_GPU if (XGB_GPU_AVAILABLE or TF_GPU_AVAILABLE) else N_ESTIMATORS
    print(f"Menggunakan {n_estimators_to_use} estimators untuk Random Forest")
    rf_model = RandomForestClassifier(n_estimators=n_estimators_to_use, 
                                     max_features=MAX_FEATURES,
                                     random_state=42,
                                     n_jobs=-1)  # Gunakan semua CPU cores
    
    # 2. Train Gradient Boosting
    print("Training Gradient Boosting...")
    # GB tidak mendukung GPU secara native, tetapi kita bisa meningkatkan kompleksitas jika ada GPU
    gb_model = GradientBoostingClassifier(n_estimators=n_estimators_to_use, 
                                         learning_rate=0.1,
                                         max_depth=5,
                                         random_state=42)
    
    # 3. Train XGBoost dengan GPU atau CPU
    print("Training XGBoost...")
    if XGB_GPU_AVAILABLE:
        print("Menggunakan GPU acceleration untuk XGBoost")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            tree_method='gpu_hist',  # Metode untuk GPU
            gpu_id=0                # GPU ID (biasanya 0 untuk GPU pertama)
        )
    else:
        print("Menggunakan CPU untuk XGBoost")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
    # 4. Train SVM jika dataset tidak terlalu besar
    if len(X_train) < 10000:  # SVM is computationally expensive for large datasets
        print("Training SVM...")
        svm_model = SVC(probability=True, 
                       C=1.0,
                       kernel='rbf', 
                       gamma='scale',
                       class_weight='balanced',
                       random_state=42,
                       max_iter=1000)
        
    # Train models dengan cross-validation
    if use_cv:
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        cv_scores = {
            'rf': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
            'gb': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
            'xgb': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
            'svm': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} if len(X_train) < 10000 else None,
            'ensemble': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        }
        
        # Train dan evaluate each model with cross-validation
        for train_idx, test_idx in tscv.split(X_scaled_df):
            X_cv_train, X_cv_test = X_scaled_df.iloc[train_idx], X_scaled_df.iloc[test_idx]
            y_cv_train, y_cv_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Random Forest
            rf_model.fit(X_cv_train, y_cv_train)
            rf_pred = rf_model.predict(X_cv_test)
            rf_proba = rf_model.predict_proba(X_cv_test)[:, 1]
            
            # Gradient Boosting
            gb_model.fit(X_cv_train, y_cv_train)
            gb_pred = gb_model.predict(X_cv_test)
            gb_proba = gb_model.predict_proba(X_cv_test)[:, 1]
            
            # XGBoost
            xgb_model.fit(X_cv_train, y_cv_train)
            xgb_pred = xgb_model.predict(X_cv_test)
            xgb_proba = xgb_model.predict_proba(X_cv_test)[:, 1]
            
            # SVM (if applicable)
            if len(X_train) < 10000:
                svm_model.fit(X_cv_train, y_cv_train)
                svm_pred = svm_model.predict(X_cv_test)
                svm_proba = svm_model.predict_proba(X_cv_test)[:, 1]
            
            # Simple ensemble (weighted average of probabilities)
            if len(X_train) < 10000:
                ensemble_proba = (rf_proba*0.3 + gb_proba*0.3 + xgb_proba*0.3 + svm_proba*0.1)
            else:
                ensemble_proba = (rf_proba*0.4 + gb_proba*0.3 + xgb_proba*0.3)
            
            ensemble_pred = (ensemble_proba > 0.5).astype(int)
            
            # Calculate metrics for each model
            for model_name, pred in [('rf', rf_pred), ('gb', gb_pred), ('xgb', xgb_pred), 
                                    ('svm', svm_pred) if len(X_train) < 10000 else None,
                                    ('ensemble', ensemble_pred)]:
                if model_name is None:
                    continue
                
                # Jika prediksi mengandung semua nilai yang sama, perhitungan metrik bisa menghasilkan nan
                # Periksa apakah prediksi memiliki variasi
                unique_preds = np.unique(pred)
                if len(unique_preds) < 2:
                    # Jika semua prediksi sama, catat metrik yang sesuai
                    if unique_preds[0] == 1:  # Semua prediksi positif
                        acc = np.mean(y_cv_test == 1)  # Akurasi = proporsi positif sebenarnya
                        prec = 1.0 if np.any(y_cv_test == 1) else 0.0  # Presisi = 1 jika ada positif sebenarnya
                        rec = 1.0  # Recall = 1 karena semua positif diprediksi dengan benar
                        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    else:  # Semua prediksi negatif
                        acc = np.mean(y_cv_test == 0)  # Akurasi = proporsi negatif sebenarnya
                        prec = 0.0  # Precision = 0 karena tidak ada positif yang diprediksi
                        rec = 0.0  # Recall = 0 karena tidak ada positif yang diprediksi
                        f1 = 0.0
                else:
                    # Jika prediksi bervariasi, hitung metrik seperti biasa
                    acc = accuracy_score(y_cv_test, pred)
                    # Gunakan zero_division=0 untuk menghindari warning
                    prec = precision_score(y_cv_test, pred, zero_division=0)
                    rec = recall_score(y_cv_test, pred, zero_division=0)
                    f1 = f1_score(y_cv_test, pred, zero_division=0)
                
                cv_scores[model_name]['accuracy'].append(acc)
                cv_scores[model_name]['precision'].append(prec)
                cv_scores[model_name]['recall'].append(rec)
                cv_scores[model_name]['f1'].append(f1)
        
        # Print hasil cross-validation untuk setiap model
        print("\n----- Hasil Cross-Validation Model Ensemble -----")
        for model_name, scores in cv_scores.items():
            if scores is None:
                continue
            
            print(f"\n{model_name.upper()} Model:")
            print(f"Accuracy: {np.mean(scores['accuracy'])*100:.2f}% (±{np.std(scores['accuracy'])*100:.2f}%)")
            print(f"Precision: {np.mean(scores['precision'])*100:.2f}% (±{np.std(scores['precision'])*100:.2f}%)")
            print(f"Recall: {np.mean(scores['recall'])*100:.2f}% (±{np.std(scores['recall'])*100:.2f}%)")
            print(f"F1 Score: {np.mean(scores['f1'])*100:.2f}% (±{np.std(scores['f1'])*100:.2f}%)")
    
    # Final training on all data
    print("\nFinal training on all data...")
    rf_model.fit(X_scaled_df, y)
    gb_model.fit(X_scaled_df, y)
    xgb_model.fit(X_scaled_df, y)
    if len(X_train) < 10000:
        svm_model.fit(X_scaled_df, y)
    
    # Store models
    models['rf'] = rf_model
    models['gb'] = gb_model
    models['xgb'] = xgb_model
    if len(X_train) < 10000:
        models['svm'] = svm_model
    
    # Feature importance (dari Random Forest)
    feature_imp = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n----- Feature Importance (Top 15) -----")
    print(feature_imp.head(15))
    
    # === Train LSTM jika dataset cukup besar dan TensorFlow tersedia ===
    try:
        if len(X_train) >= 1000 and TENSORFLOW_AVAILABLE:  # LSTM needs sufficient data
            print("\nTraining LSTM neural network...")
            
            # Persiapan data untuk LSTM (sequence)
            sequence_length = 10  # Window size for sequence
            X_lstm, y_lstm = create_sequences(X_scaled_df.values, y.values, sequence_length)
            
            # Split data for LSTM
            X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(
                X_lstm, y_lstm, test_size=0.2, shuffle=False)
            
            # Build LSTM model
            lstm_model = build_lstm_model(X_lstm.shape[1:], len(features))
            
            if lstm_model is not None:
                # Early stopping
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                # Train LSTM
                lstm_history = lstm_model.fit(
                    X_lstm_train, y_lstm_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_lstm_test, y_lstm_test),
                    callbacks=[early_stop],
                    verbose=1
                )
                
                # Evaluate LSTM
                lstm_eval = lstm_model.evaluate(X_lstm_test, y_lstm_test, verbose=0)
                print(f"LSTM Test Loss: {lstm_eval[0]:.4f}, Test Accuracy: {lstm_eval[1]*100:.2f}%")
                
                # Store LSTM model
                models['lstm'] = lstm_model
                
                # Save sequence scaler info for prediction
                models['lstm_seq_length'] = sequence_length
        elif not TENSORFLOW_AVAILABLE:
            print("Melewati training LSTM karena TensorFlow tidak tersedia")
    except Exception as e:
        print(f"Error training LSTM: {e}")
        print("Continuing without LSTM model")
    
    # Save model jika diperlukan
    if SAVE_MODEL:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{OUTPUT_DIR}/models_{PAIR.replace('/','_')}_{TIMEFRAME}_{timestamp}.joblib"
        joblib.dump(models, model_filename)
        print(f"Models disimpan ke: {model_filename}")
        
        # Simpan juga scaler dan feature info
        joblib.dump(scaler, f"{OUTPUT_DIR}/scaler_{PAIR.replace('/','_')}_{TIMEFRAME}_{timestamp}.joblib")
        feature_imp.to_csv(f"{OUTPUT_DIR}/feature_importance_{PAIR.replace('/','_')}_{TIMEFRAME}_{timestamp}.csv")
        # Simpan fitur ke file agar bisa di-load saat live
        joblib.dump(features, f"{OUTPUT_DIR}/features_{PAIR.replace('/','_')}_{TIMEFRAME}_{timestamp}.joblib")
    
    return models, features, feature_imp, scaler

def create_sequences(X, y, seq_length):
    """Create sequences for LSTM"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(input_shape, n_features):
    """Build LSTM model architecture"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow tidak tersedia, model LSTM tidak dapat dibuat")
        return None
    
    # Periksa ketersediaan GPU dan sesuaikan arsitektur
    if TF_GPU_AVAILABLE:
        print("Membangun model LSTM dengan optimasi GPU")
        # Arsitektur yang lebih besar dan kompleks untuk GPU
        model = Sequential()
        model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        
        model.add(LSTM(units=32))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        # Gunakan Adam optimizer dengan learning rate yang optimal
        optimizer = Adam(learning_rate=0.001)
    else:
        print("Membangun model LSTM untuk CPU (lebih sederhana)")
        # Arsitektur yang lebih sederhana untuk CPU
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        model.add(LSTM(units=32))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        # Gunakan Adam optimizer dengan learning rate default
        optimizer = Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def generate_signals_ensemble(df, models, features, scaler):
    """Generate trading signals using ensemble model predictions"""
    # Prepare features - pastikan semua fitur yang diperlukan tersedia
    required_features = features.copy()
    
    # Periksa fitur yang hilang dan tambahkan dengan nilai default (0)
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        print(f"Menambahkan {len(missing_features)} fitur yang hilang dengan nilai 0")
        for feature in missing_features:
            df[feature] = 0
    
    # Pilih hanya fitur yang diperlukan model dalam urutan yang sama dengan training
    X = df[required_features]
    
    # Pastikan jumlah fitur cocok dengan model
    if X.shape[1] != len(required_features):
        print(f"WARNING: Jumlah fitur tidak cocok! Model membutuhkan {len(required_features)}, data memiliki {X.shape[1]}")
        # Coba sesuaikan jika memungkinkan
        for f in required_features:
            if f not in X.columns:
                print(f"Fitur hilang: {f}")
    
    try:
        # Scale data
        X_scaled = scaler.transform(X)
        
        # Get predictions from each model
        preds = {}
        probas = {}
        
        # Get predictions from classical models
        for model_name, model in models.items():
            if model_name != 'lstm' and model_name != 'lstm_seq_length':
                try:
                    preds[model_name] = model.predict(X_scaled)
                    probas[model_name] = model.predict_proba(X_scaled)[:, 1]
                except Exception as e:
                    print(f"Error predicting with {model_name}: {e}")
                    # Gunakan prediksi default jika model gagal
                    preds[model_name] = np.ones(len(X)) * 0.5
                    probas[model_name] = np.ones(len(X)) * 0.5
        
        # Get LSTM predictions if available
        if TENSORFLOW_AVAILABLE and 'lstm' in models and 'lstm_seq_length' in models:
            seq_length = models['lstm_seq_length']
            lstm_model = models['lstm']
            
            # Create sequences for LSTM prediction
            X_lstm = []
            for i in range(len(X_scaled) - seq_length):
                X_lstm.append(X_scaled[i:i + seq_length])
            
            if X_lstm:
                X_lstm = np.array(X_lstm)
                try:
                    lstm_pred = lstm_model.predict(X_lstm)
                    
                    # Pad with NaN for the first seq_length rows
                    lstm_full_pred = np.full(len(X_scaled), np.nan)
                    lstm_full_pred[seq_length:] = (lstm_pred.flatten() > 0.5).astype(int)
                    lstm_full_proba = np.full(len(X_scaled), np.nan)
                    lstm_full_proba[seq_length:] = lstm_pred.flatten()
                    
                    preds['lstm'] = lstm_full_pred
                    probas['lstm'] = lstm_full_proba
                except Exception as e:
                    print(f"Error predicting with LSTM: {e}")
        
        # Calculate weighted ensemble prediction
        weights = {
            'rf': 0.3,
            'gb': 0.3,
            'xgb': 0.3,
            'svm': 0.1,
            'lstm': 0.3  # Higher weight for LSTM if available
        }
        
        # Adjust weights if some models are missing
        if 'svm' not in probas:
            weights['rf'] = 0.4
            weights['gb'] = 0.3
            weights['xgb'] = 0.3
        
        if 'lstm' in probas:
            # Rescale classical model weights
            classical_weight_sum = weights['rf'] + weights['gb'] + weights['xgb']
            if 'svm' in weights:
                classical_weight_sum += weights['svm']
            
            lstm_weight = weights['lstm']
            rescale_factor = (1 - lstm_weight) / classical_weight_sum
            
            for model in ['rf', 'gb', 'xgb', 'svm']:
                if model in weights:
                    weights[model] = weights[model] * rescale_factor
        
        # Calculate weighted probabilities
        ensemble_proba = np.zeros(len(X))
        weight_sum = 0
        
        for model_name, proba in probas.items():
            if model_name in weights:
                # Handle NaN values (e.g., from LSTM)
                valid_mask = ~np.isnan(proba)
                ensemble_proba[valid_mask] += proba[valid_mask] * weights[model_name]
                weight_sum += weights[model_name] * np.mean(valid_mask)
        
        # Normalize by weight sum
        if weight_sum > 0:
            ensemble_proba = ensemble_proba / weight_sum
        
        # Convert to binary prediction
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        # Generate signals
        df['pred'] = ensemble_pred
        df['pred_proba'] = ensemble_proba
        df['returns'] = df['close'].pct_change().shift(-1)
        df['strategy'] = df['returns'] * df['pred']
        df['signal'] = df['pred'].map({1: 'buy', 0: 'hold'})
        
        # Add individual model predictions for analysis
        for model_name, pred in preds.items():
            if not np.all(np.isnan(pred)):
                df[f'pred_{model_name}'] = pred
        
        return df
    except Exception as e:
        print(f"Error dalam generate signals: {e}")
        # Buat default signals untuk mencegah crash
        df['pred'] = 0
        df['pred_proba'] = 0.5
        df['returns'] = df['close'].pct_change().shift(-1)
        df['strategy'] = 0
        df['signal'] = 'hold'
        return df

def run_backtest(df, initial_capital=11):
    """
    Jalankan backtest dengan data yang sudah disiapkan.
    Return: tuple (df_trades, df_positions, equity_curve)
    """
    capital = initial_capital
    trades = []
    positions = []
    current_position = None
    equity_curve = [{'date': df.index[0], 'equity': capital}]
    
    print("\n----- Simulasi Backtest dengan Fee Trading -----")
    for i in range(len(df)-1):
        current_row = df.iloc[i]
        next_row = df.iloc[i+1]
        
        # Jika tidak punya posisi dan sinyal beli
        if current_position is None and current_row['signal'] == 'buy':
            # Hitung jumlah yang bisa dibeli (dengan fee)
            entry_price = next_row['open']  # Beli di open candle berikutnya
            position_size = min(TRADE_AMOUNT, capital)
            fee = position_size * TRADING_FEE
            actual_position = position_size - fee
            qty = actual_position / entry_price
            
            # Catat posisi
            current_position = {
                'entry_date': next_row.name,
                'entry_price': entry_price,
                'position_size': actual_position,
                'qty': qty,
                'stop_loss': entry_price * (1 - STOP_LOSS_PCT),
                'take_profit': entry_price * (1 + TAKE_PROFIT_PCT)
            }
            
            # Update capital
            capital -= position_size
            trades.append({
                'date': next_row.name,
                'action': 'BUY',
                'price': entry_price,
                'qty': qty,
                'value': position_size,
                'fee': fee,
                'capital': capital
            })
        
        # Jika punya posisi, cek exit (take profit atau stop loss)
        elif current_position is not None:
            # Cek harga untuk SL/TP (gunakan high/low untuk simulasi lebih realistis)
            high_price = next_row['high']
            low_price = next_row['low']
            close_price = next_row['close']
            
            # Hitung potensi profit/loss
            exit_reason = None
            exit_price = None
            
            # Cek take profit
            if high_price >= current_position['take_profit']:
                exit_reason = 'TP'
                exit_price = current_position['take_profit']
            # Cek stop loss
            elif low_price <= current_position['stop_loss']:
                exit_reason = 'SL'
                exit_price = current_position['stop_loss']
            # Cek sinyal exit (pred = 0)
            elif current_row['signal'] == 'hold':
                exit_reason = 'Signal'
                exit_price = next_row['open']  # Exit di open candle berikutnya
            
            # Jika ada alasan exit
            if exit_reason:
                # Hitung hasil exit
                exit_value = current_position['qty'] * exit_price
                fee = exit_value * TRADING_FEE
                net_value = exit_value - fee
                
                # Update capital
                capital += net_value
                
                # Catat trade
                positions.append({
                    'entry_date': current_position['entry_date'],
                    'exit_date': next_row.name,
                    'entry_price': current_position['entry_price'],
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl': net_value - current_position['position_size'],
                    'pnl_pct': (net_value / current_position['position_size'] - 1) * 100
                })
                
                trades.append({
                    'date': next_row.name,
                    'action': 'SELL',
                    'price': exit_price,
                    'qty': current_position['qty'],
                    'value': exit_value,
                    'fee': fee,
                    'capital': capital,
                    'reason': exit_reason
                })
                
                # Reset posisi
                current_position = None
        
        # Catat equity curve di setiap candle
        equity_curve.append({
            'date': next_row.name,
            'equity': capital + (current_position['qty'] * next_row['close'] if current_position else 0)
        })
    
    # Buat DataFrame untuk analisis trades
    trades_df = pd.DataFrame(trades)
    positions_df = pd.DataFrame(positions)
    equity_df = pd.DataFrame(equity_curve)
    equity_df.set_index('date', inplace=True)
    
    return trades_df, positions_df, equity_df

def plot_equity_curve(equity_df, pair, timeframe):
    """
    Plot equity curve dan simpan hasilnya
    """
    if not PLOT_RESULTS:
        return
    
    # Get HODL performance for comparison
    try:
        # Get data for the same period as equity curve
        start_date = equity_df.index[0]
        end_date = equity_df.index[-1]
        
        # Fetch data
        df_market = get_data_ccxt(symbol=pair, timeframe=timeframe, limit=DATA_LIMIT)
        df_market = df_market.loc[(df_market.index >= start_date) & (df_market.index <= end_date)]
        
        if not df_market.empty:
            # Calculate HODL equity curve (scaled to same initial capital)
            initial_capital = equity_df['equity'].iloc[0]
            initial_price = df_market['close'].iloc[0]
            
            df_market['hodl_equity'] = df_market['close'] / initial_price * initial_capital
            
            # Merge with equity curve (only keep dates in equity_df)
            hodl_equity = df_market['hodl_equity'].reindex(equity_df.index, method='ffill')
            equity_df['hodl_equity'] = hodl_equity
    except Exception as e:
        print(f"Error calculating HODL comparison: {e}")
        # Continue without HODL comparison
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plotting main equity curve
    plt.subplot(2, 1, 1)
    plt.plot(equity_df.index, equity_df['equity'], label='Bot Strategy', color='blue', linewidth=2)
    
    # Add HODL curve if available
    if 'hodl_equity' in equity_df.columns:
        plt.plot(equity_df.index, equity_df['hodl_equity'], label='HODL Strategy', color='red', linestyle='--', linewidth=1.5)
    
    # Add reference line and details
    plt.axhline(y=equity_df['equity'].iloc[0], color='gray', linestyle=':', label='Initial Capital')
    plt.title(f'Equity Curve Comparison - {pair} {timeframe}', fontsize=14)
    plt.ylabel('Equity ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plotting relative performance (% difference)
    if 'hodl_equity' in equity_df.columns:
        plt.subplot(2, 1, 2)
        
        # Calculate outperformance percentage
        equity_df['outperformance'] = ((equity_df['equity'] / equity_df['hodl_equity']) - 1) * 100
        
        # Plot outperformance line
        plt.plot(equity_df.index, equity_df['outperformance'], color='green', label='Bot vs HODL (%)')
        plt.axhline(y=0, color='gray', linestyle=':', label='Equal Performance')
        
        # Mark areas of outperformance/underperformance
        plt.fill_between(equity_df.index, equity_df['outperformance'], 0, 
                         where=(equity_df['outperformance'] >= 0),
                         color='green', alpha=0.2, label='Outperformance')
        plt.fill_between(equity_df.index, equity_df['outperformance'], 0, 
                         where=(equity_df['outperformance'] < 0),
                         color='red', alpha=0.2, label='Underperformance')
        
        plt.title('Bot Outperformance vs HODL (%)', fontsize=14)
        plt.ylabel('Outperformance (%)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Simpan plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{OUTPUT_DIR}/equity_curve_{pair.replace('/','_')}_{timeframe}_{timestamp}.png", dpi=300)
    plt.close()

def plot_trade_analysis(positions_df, pair, timeframe):
    """
    Plot analisis trades
    """
    if not PLOT_RESULTS or len(positions_df) < 1:
        return
    
    # 1. Distribusi profit/loss
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    sns.histplot(positions_df['pnl_pct'], kde=True)
    plt.title('Distribusi Profit/Loss (%)')
    plt.axvline(x=0, color='r', linestyle='--')
    
    # 2. Profit/loss berdasarkan waktu
    plt.subplot(2, 2, 2)
    positions_df['exit_date'] = pd.to_datetime(positions_df['exit_date'])
    positions_df.set_index('exit_date', inplace=True)
    positions_df['pnl_pct'].plot()
    plt.title('Profit/Loss per Trade')
    plt.axhline(y=0, color='r', linestyle='--')
    
    # 3. Profit/loss berdasarkan alasan exit
    plt.subplot(2, 2, 3)
    if 'exit_reason' in positions_df.columns:
        sns.boxplot(x='exit_reason', y='pnl_pct', data=positions_df.reset_index())
        plt.title('Profit/Loss by Exit Reason')
    
    # 4. Cumulative returns
    plt.subplot(2, 2, 4)
    positions_df['pnl_pct'].cumsum().plot()
    plt.title('Cumulative Returns (%)')
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.tight_layout()
    
    # Simpan plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{OUTPUT_DIR}/trade_analysis_{pair.replace('/','_')}_{timeframe}_{timestamp}.png")
    plt.close()

def print_backtest_results(positions_df, initial_capital, final_capital, pair, timeframe):
    """
    Print hasil backtest
    """
    profit = final_capital - initial_capital
    profit_pct = (final_capital / initial_capital - 1) * 100
    
    # Hitung total fee yang dibayarkan
    total_fee = 0
    entry_fee = 0
    exit_fee = 0
    
    if len(positions_df) > 0:
        win_trades = len(positions_df[positions_df['pnl'] > 0])
        loss_trades = len(positions_df[positions_df['pnl'] <= 0])
        total_trades = len(positions_df)
        win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
        
        avg_profit = positions_df['pnl_pct'].mean()
        max_profit = positions_df['pnl_pct'].max() 
        max_loss = positions_df['pnl_pct'].min()
        
        profit_factor = abs(positions_df[positions_df['pnl'] > 0]['pnl'].sum() / 
                           positions_df[positions_df['pnl'] < 0]['pnl'].sum()) if len(positions_df[positions_df['pnl'] < 0]) > 0 else float('inf')
        
        # Summary by exit reason
        if 'exit_reason' in positions_df.columns:
            reason_summary = positions_df.groupby('exit_reason').agg({
                'pnl': ['count', 'mean', 'sum'],
                'pnl_pct': ['mean', 'min', 'max']
            })
            
        # Calculate total fees - assuming entry size is TRADE_AMOUNT per trade
        entry_fee = total_trades * TRADE_AMOUNT * TRADING_FEE
        exit_fee = total_trades * TRADE_AMOUNT * (1 + avg_profit/100) * TRADING_FEE
        total_fee = entry_fee + exit_fee
        fee_pct = (total_fee / initial_capital) * 100
        
        # Calculate net profit after fees (already included in backtest but showing separately)
        net_profit = profit
        net_profit_pct = profit_pct
        
        # Calculate HODL return from first entry to last exit
        if len(positions_df) > 1:
            # Get historical data for the HODL period
            first_entry = pd.to_datetime(positions_df['entry_date'].iloc[0])
            last_exit = pd.to_datetime(positions_df['exit_date'].iloc[-1])
            
            # Fetch data from exchange
            df_hodl = get_data_ccxt(symbol=pair, timeframe=timeframe, limit=DATA_LIMIT)
            
            # Find closest timestamps
            df_hodl = df_hodl.loc[df_hodl.index >= first_entry]
            
            if not df_hodl.empty:
                first_price = df_hodl['close'].iloc[0]
                # Find the closest date to last_exit that exists in df_hodl
                last_idx = df_hodl.index.get_indexer([last_exit], method='nearest')[0]
                last_price = df_hodl['close'].iloc[last_idx]
                
                hodl_return = ((last_price / first_price) - 1) * 100
            else:
                hodl_return = 0
        else:
            hodl_return = 0
    else:
        win_trades = loss_trades = total_trades = 0
        win_rate = avg_profit = max_profit = max_loss = profit_factor = 0
        hodl_return = 0
        total_fee = entry_fee = exit_fee = fee_pct = 0
        net_profit = profit
        net_profit_pct = profit_pct
    
    print("\n----- Hasil Backtest -----")
    print(f"Pair: {pair} | Timeframe: {timeframe} | Target Threshold: {TARGET_THRESHOLD*100}%")
    print(f"Modal Awal: ${initial_capital}")
    print(f"Modal Akhir: ${final_capital:.2f}")
    print(f"Profit/Loss: ${profit:.2f} ({profit_pct:.2f}%)")
    print(f"Total Trades: {total_trades}")
    
    if total_trades > 0:
        print(f"Win Rate: {win_rate:.2f}% ({win_trades}/{total_trades})")
        print(f"Avg Profit: {avg_profit:.2f}%")
        print(f"Max Profit: {max_profit:.2f}%")
        print(f"Max Loss: {max_loss:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        # Fee analysis
        print("\n----- Fee Analysis -----")
        print(f"Total Fee: ${total_fee:.2f} ({fee_pct:.2f}% dari modal)")
        print(f"Entry Fee: ${entry_fee:.2f}")
        print(f"Exit Fee: ${exit_fee:.2f}")
        print(f"Net Profit After Fee: ${net_profit:.2f} ({net_profit_pct:.2f}%)")
        
        # HODL comparison
        print("\n----- HODL Comparison -----")
        print(f"HODL Return: {hodl_return:.2f}%")
        outperformance = net_profit_pct - hodl_return
        print(f"Bot vs HODL: {outperformance:+.2f}% {'(OUTPERFORM)' if outperformance > 0 else '(UNDERPERFORM)'}")
        
        # Market condition analysis
        if abs(hodl_return) < 1:
            market_condition = "SIDEWAYS"
        elif hodl_return > 0:
            market_condition = "UPTREND"
        else:
            market_condition = "DOWNTREND"
        print(f"Market Condition: {market_condition}")
        
        if 'exit_reason' in positions_df.columns:
            print("\n----- Exit Reason Analysis -----")
            print(reason_summary)
            
        # Kirim ringkasan hasil ke Telegram
        emoji_profit = "🟢" if profit_pct > 0 else "🔴"
        emoji_market = "📈" if hodl_return > 0 else "📉" if hodl_return < 0 else "➡️"
        emoji_comp = "🚀" if outperformance > 0 else "🐌"
        
        backtest_msg = (
            f"📊 <b>HASIL BACKTEST</b>\n\n"
            f"Pair: {pair}\n"
            f"Timeframe: {timeframe}\n"
            f"Market: {market_condition} {emoji_market}\n\n"
            f"{emoji_profit} P/L: {profit_pct:.2f}%\n"
            f"💰 Modal awal: ${initial_capital}\n"
            f"💸 Modal akhir: ${final_capital:.2f}\n\n"
            f"🎯 Win Rate: {win_rate:.2f}%\n"
            f"📋 Total trades: {total_trades}\n"
            f"📊 Profit factor: {profit_factor:.2f}\n\n"
            f"🆚 HODL: {hodl_return:.2f}%\n"
            f"{emoji_comp} Outperform: {outperformance:+.2f}%\n\n"
            f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        send_telegram(backtest_msg)
    
    # Simpan hasil ke CSV jika diminta
    if SAVE_RESULTS:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(positions_df) > 0:
            positions_df.to_csv(f"{OUTPUT_DIR}/positions_{pair.replace('/','_')}_{timeframe}_{timestamp}.csv")
        
        # Simpan summary ke file terpisah
        summary = {
            'pair': pair,
            'timeframe': timeframe,
            'threshold': TARGET_THRESHOLD,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'profit': profit,
            'profit_pct': profit_pct,
            'total_trades': total_trades,
            'win_rate': win_rate if total_trades > 0 else 0,
            'avg_profit': avg_profit if total_trades > 0 else 0,
            'max_profit': max_profit if total_trades > 0 else 0,
            'max_loss': max_loss if total_trades > 0 else 0,
            'profit_factor': profit_factor if total_trades > 0 else 0,
            'total_fee': total_fee,
            'hodl_return': hodl_return,
            'outperformance': net_profit_pct - hodl_return if total_trades > 0 else 0,
            'market_condition': market_condition if total_trades > 0 else 'UNKNOWN',
            'timestamp': timestamp
        }
        
        # Cek apakah file summary sudah ada
        summary_file = f"{OUTPUT_DIR}/backtest_summary.csv"
        if os.path.exists(summary_file):
            summary_df = pd.read_csv(summary_file)
            summary_df = pd.concat([summary_df, pd.DataFrame([summary])], ignore_index=True)
        else:
            summary_df = pd.DataFrame([summary])
        
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Hasil backtest disimpan ke {OUTPUT_DIR}")

def get_multi_timeframe_data(symbol=PAIR, base_timeframe=TIMEFRAME, limit=DATA_LIMIT):
    """
    Mendapatkan data dari berbagai timeframe untuk analisis multi-timeframe
    """
    # Timeframes yang lebih tinggi untuk dianalisis
    higher_timeframes = []
    
    # Menentukan timeframe yang lebih tinggi berdasarkan base_timeframe
    if base_timeframe == '1m':
        higher_timeframes = ['5m', '15m', '1h', '4h']
    elif base_timeframe == '5m':
        higher_timeframes = ['15m', '1h', '4h', '1d']
    elif base_timeframe == '15m':
        higher_timeframes = ['1h', '4h', '1d']
    elif base_timeframe == '1h':
        higher_timeframes = ['4h', '1d']
    elif base_timeframe == '4h':
        higher_timeframes = ['1d']
    else:
        higher_timeframes = ['1d']
        
    # Pilih 2 timeframe teratas saja untuk efisiensi
    higher_timeframes = higher_timeframes[:2]
    
    print(f"Mengambil data multi-timeframe: {higher_timeframes}")
    
    # Ambil data untuk setiap timeframe
    multi_tf_data = {}
    ex = ccxt.binance({'enableRateLimit': True})
    
    for tf in higher_timeframes:
        try:
            # Mengambil data dengan jumlah candle lebih sedikit untuk timeframe yang lebih tinggi
            tf_limit = max(100, limit // 2)
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=tf_limit)
            df_tf = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_tf['timestamp'] = pd.to_datetime(df_tf['timestamp'], unit='ms')
            df_tf.set_index('timestamp', inplace=True)
            multi_tf_data[tf] = df_tf
        except Exception as e:
            print(f"Error mengambil data timeframe {tf}: {e}")
            multi_tf_data[tf] = pd.DataFrame()
    
    return multi_tf_data

# ========== MAIN FUNCTION ==========
def run_strategy(pair=PAIR, timeframe=TIMEFRAME, data_limit=DATA_LIMIT, initial_capital=11):
    """
    Fungsi utama untuk menjalankan strategi
    """
    print(f"\n{'='*20} RUNNING STRATEGY: {pair} {timeframe} {'='*20}")
    
    # Kirim notifikasi Telegram saat memulai strategi
    send_telegram(f"🚀 BOT TRADING DIMULAI\nPair: {pair}\nTimeframe: {timeframe}\nMode: {'LIVE' if IS_LIVE else 'SIMULASI'}")
    
    # Step 1: Load data dari berbagai timeframe
    print("Loading data multi-timeframe...")
    df = get_data_ccxt(symbol=pair, timeframe=timeframe, limit=data_limit)
    
    # Get data dari timeframe yang lebih tinggi untuk analisis multi-timeframe
    multi_tf_data = get_multi_timeframe_data(symbol=pair, base_timeframe=timeframe, limit=data_limit)
    
    # Step 2: Feature engineering dengan data multi-timeframe
    print("Memproses indikator teknikal...")
    df = feature_engineering(df, multi_tf_data)
    df = label_target(df, threshold=TARGET_THRESHOLD)
    
    # Step 3: Training model ensemble dan deep learning
    print("Training model ensemble dan deep learning...")
    models, features, feature_imp, scaler = train_ml_advanced(df, use_cv=True)
    
    # Step 4: Generate signals menggunakan ensemble prediction
    df = generate_signals_ensemble(df, models, features, scaler)
    trades_df, positions_df, equity_df = run_backtest(df, initial_capital)
    
    # Analisis hasil
    final_capital = equity_df['equity'].iloc[-1]
    print_backtest_results(positions_df, initial_capital, final_capital, pair, timeframe)
    
    # Plot hasil
    plot_equity_curve(equity_df, pair, timeframe)
    plot_trade_analysis(positions_df, pair, timeframe)
    
    return models, features, feature_imp, scaler, df, trades_df, positions_df, equity_df

# ========== SIMULASI/BACKTEST ==========
# Cek koneksi Telegram terlebih dahulu
print("\n----- Cek Koneksi Telegram -----")
telegram_connected = check_telegram_connection()
print(f"Status Telegram: {'TERHUBUNG' if telegram_connected else 'GAGAL TERHUBUNG'}")

# Jalankan strategi dengan parameter default
models, features, feature_imp, scaler, df, trades_df, positions_df, equity_df = run_strategy()

# ========== 2. UPGRADE KE BOT TRADING REAL-TIME ==========
if IS_LIVE:
    binance = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
    })

    # Kirim notifikasi Telegram mode live
    send_telegram(f"🔴 <b>BOT TRADING LIVE DIAKTIFKAN</b>\n\nPair: {PAIR}\nTimeframe: {TIMEFRAME}\nModal: ${TRADE_AMOUNT}\nStop Loss: {STOP_LOSS_PCT*100}%\nTake Profit: {TAKE_PROFIT_PCT*100}%")

    # Load model dari file jika tersedia
    model_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"models_{PAIR.replace('/','_')}_{TIMEFRAME}") and f.endswith(".joblib")]
    scaler_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"scaler_{PAIR.replace('/','_')}_{TIMEFRAME}") and f.endswith(".joblib")]
    features_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"features_{PAIR.replace('/','_')}_{TIMEFRAME}") and f.endswith(".joblib")]

    if model_files and scaler_files and os.path.exists(f"{OUTPUT_DIR}/{model_files[-1]}") and os.path.exists(f"{OUTPUT_DIR}/{scaler_files[-1]}"):
        latest_model = model_files[-1]
        latest_scaler = scaler_files[-1]
        try:
            print(f"Loading model dari {latest_model}...")
            models = joblib.load(f"{OUTPUT_DIR}/{latest_model}")
            scaler = joblib.load(f"{OUTPUT_DIR}/{latest_scaler}")
            # Load fitur dari file jika ada
            if features_files:
                features = joblib.load(f"{OUTPUT_DIR}/{features_files[-1]}")
                print(f"Fitur loaded dari file: {len(features)} fitur")
            else:
                # Fallback: load dari feature importance
                imp_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"feature_importance_{PAIR.replace('/','_')}_{TIMEFRAME}") and f.endswith(".csv")]
                if imp_files:
                    feature_imp = pd.read_csv(f"{OUTPUT_DIR}/{imp_files[-1]}", index_col=0)
                    features = select_features(None, feature_imp)
                    print(f"Menggunakan {len(features)} fitur teratas dari feature importance")
                else:
                    print("Tidak menemukan file fitur, training ulang untuk mendapatkan fitur")
                    df_temp = get_data_ccxt(symbol=PAIR, timeframe=TIMEFRAME, limit=200)
                    multi_tf_data = get_multi_timeframe_data(symbol=PAIR, base_timeframe=TIMEFRAME, limit=200)
                    df_temp = feature_engineering(df_temp, multi_tf_data)
                    df_temp = label_target(df_temp)
                    features = select_features(df_temp)
                    print(f"Menggunakan {len(features)} fitur dari data training baru")
            # Validasi model dan scaler
            df_test = get_data_ccxt(symbol=PAIR, timeframe=TIMEFRAME, limit=30)
            df_test = feature_engineering(df_test, None)
            for f in features:
                if f not in df_test.columns:
                    df_test[f] = 0
            df_test = df_test.dropna()
            if len(df_test) == 0:
                print('Data untuk validasi model kosong, skip validasi.')
            else:
                test_X = df_test[features]
                test_X_scaled = scaler.transform(test_X)
                for model_name, model in models.items():
                    if model_name != 'lstm' and model_name != 'lstm_seq_length':
                        _ = model.predict_proba(test_X_scaled)
            print("Semua model berhasil divalidasi!")
        except Exception as e:
            print(f"Error loading/validating model: {e}")
            send_telegram(f"⚠️ <b>Error Loading Model</b>\n{PAIR} {TIMEFRAME}\n{str(e)}\nMemulai training baru...")
            df = get_data_ccxt(symbol=PAIR, timeframe=TIMEFRAME, limit=DATA_LIMIT)
            multi_tf_data = get_multi_timeframe_data(symbol=PAIR, base_timeframe=TIMEFRAME, limit=DATA_LIMIT)
            df = feature_engineering(df, multi_tf_data)
            df = label_target(df)
            models, features, feature_imp, scaler = train_ml_advanced(df, use_cv=True)
    else:
        print("Tidak ada model tersimpan, training model baru...")
        df = get_data_ccxt(symbol=PAIR, timeframe=TIMEFRAME, limit=DATA_LIMIT)
        multi_tf_data = get_multi_timeframe_data(symbol=PAIR, base_timeframe=TIMEFRAME, limit=DATA_LIMIT)
        df = feature_engineering(df, multi_tf_data)
        df = label_target(df)
        models, features, feature_imp, scaler = train_ml_advanced(df, use_cv=True)

    print(f"BOT TRADING AKTIF. Running real-time untuk {PAIR} di {TIMEFRAME}...")
    last_timestamp = None
    positions = []
    current_position = None
    trade_done = False  # Hanya lakukan 1 trading

    # Simpan state bot
    bot_state = {
        'last_timestamp': None,
        'current_position': None,
        'trades': [],
        'positions': [],
        'start_capital': 0,
        'current_capital': 0
    }
    state_file = f"{OUTPUT_DIR}/bot_state_{PAIR.replace('/','_')}_{TIMEFRAME}.joblib"
    if os.path.exists(state_file):
        try:
            bot_state = joblib.load(state_file)
            print(f"Bot state dimuat dari {state_file}")
            last_timestamp = bot_state['last_timestamp']
            current_position = bot_state['current_position']
        except Exception as e:
            print(f"Error loading bot state: {e}")

    while not trade_done:
        try:
            df = get_data_ccxt(symbol=PAIR, timeframe=TIMEFRAME, limit=DATA_LIMIT)
            multi_tf_data = get_multi_timeframe_data(symbol=PAIR, base_timeframe=TIMEFRAME, limit=100)
            df = feature_engineering(df, multi_tf_data)
            # Pastikan semua fitur tersedia dan urutannya sama
            for f in features:
                if f not in df.columns:
                    df[f] = 0
            X = df[features]
            try:
                X_scaled = scaler.transform(X)
                ensemble_proba = 0
                model_count = 0
                for model_name, model in models.items():
                    if model_name != 'lstm' and model_name != 'lstm_seq_length':
                        proba = model.predict_proba(X_scaled)[:, 1]
                        weight = 0.3 if model_name in ['rf', 'gb', 'xgb'] else 0.1
                        ensemble_proba += proba * weight
                        model_count += weight
                if model_count > 0:
                    ensemble_proba = ensemble_proba / model_count
                df['pred_proba'] = ensemble_proba
                df['pred'] = (ensemble_proba > 0.5).astype(int)
                df['signal'] = df['pred'].map({1:'buy', 0:'hold'})
            except Exception as e:
                print(f"Error saat prediksi: {str(e)}")
                df['pred_proba'] = 0.5
                df['pred'] = 0
                df['signal'] = 'hold'
            latest = df.iloc[-1]
            if last_timestamp == latest.name:
                print(f"Belum ada candle baru. Menunggu...")
            else:
                last_signal = latest['signal']
                last_price = float(latest['close'])
                last_timestamp = latest.name
                bot_state['last_timestamp'] = last_timestamp
                try:
                    balance = binance.fetch_balance()
                    free_usdt = balance['total']['USDT']
                    if bot_state['start_capital'] == 0:
                        bot_state['start_capital'] = free_usdt
                    bot_state['current_capital'] = free_usdt
                    if current_position is None and last_signal == 'buy' and free_usdt >= TRADE_AMOUNT and latest['pred_proba'] >= 0.6:
                        entry_price = last_price
                        position_size = min(TRADE_AMOUNT, free_usdt)
                        fee = position_size * TRADING_FEE
                        actual_position = position_size - fee
                        qty = actual_position / entry_price
                        stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                        take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
                        try:
                            order = binance.create_market_buy_order(PAIR, TRADE_AMOUNT/entry_price)
                            current_position = {
                                'entry_date': latest.name,
                                'entry_price': entry_price,
                                'position_size': actual_position,
                                'qty': qty,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'order_id': order['id'] if 'id' in order else None,
                                'confidence': latest['pred_proba']
                            }
                            bot_state['current_position'] = current_position
                            msg = (f"🔵 <b>BOT ENTRY BUY</b>\n"
                                   f"Pair: {PAIR}\n"
                                   f"Timeframe: {TIMEFRAME}\n"
                                   f"💰 Entry: ${entry_price:.2f}\n"
                                   f"🔢 Qty: {qty:.6f}\n"
                                   f"📈 TP: ${take_profit:.2f}\n"
                                   f"📉 SL: ${stop_loss:.2f}\n"
                                   f"🎯 Confidence: {latest['pred_proba']:.2f}\n"
                                   f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            print(msg)
                            send_telegram(msg)
                        except Exception as e:
                            print(f"[!] Order gagal: {e}")
                            send_telegram(f"⚠️ <b>Bot Order Error</b>\n{PAIR} {TIMEFRAME}\n{e}")
                    elif current_position is not None:
                        exit_reason = None
                        if last_price >= current_position['take_profit']:
                            exit_reason = 'TP'
                        elif last_price <= current_position['stop_loss']:
                            exit_reason = 'SL'
                        elif last_signal == 'hold' and (1 - latest['pred_proba']) >= 0.6:
                            exit_reason = 'Signal'
                        if exit_reason:
                            try:
                                order = binance.create_market_sell_order(PAIR, current_position['qty'])
                                exit_value = current_position['qty'] * last_price
                                fee = exit_value * TRADING_FEE
                                net_value = exit_value - fee
                                profit = net_value - current_position['position_size']
                                profit_pct = (net_value / current_position['position_size'] - 1) * 100
                                position_result = {
                                    'entry_date': current_position['entry_date'],
                                    'exit_date': latest.name,
                                    'entry_price': current_position['entry_price'],
                                    'exit_price': last_price,
                                    'exit_reason': exit_reason,
                                    'qty': current_position['qty'],
                                    'pnl': profit,
                                    'pnl_pct': profit_pct,
                                    'entry_confidence': current_position.get('confidence', 0),
                                    'exit_confidence': latest['pred_proba'] if exit_reason == 'Signal' else 0
                                }
                                positions.append(position_result)
                                bot_state['positions'].append(position_result)
                                current_position = None
                                bot_state['current_position'] = None
                                msg = (f"🔴 <b>BOT EXIT {exit_reason}</b>\n"
                                       f"Pair: {PAIR}\n"
                                       f"Timeframe: {TIMEFRAME}\n"
                                       f"💰 Exit: ${last_price:.2f}\n"
                                       f"{'🟢' if profit > 0 else '🔴'} P/L: ${profit:.2f} ({profit_pct:.2f}%)\n"
                                       f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                print(msg)
                                send_telegram(msg)
                                trade_done = True  # Selesai 1 trading, keluar loop
                            except Exception as e:
                                print(f"[!] Exit order gagal: {e}")
                                send_telegram(f"⚠️ <b>Bot Exit Error</b>\n{PAIR} {TIMEFRAME}\n{e}")
                except Exception as e:
                    print(f"[!!] Error balance: {e}")
                    send_telegram(f"⚠️ <b>Bot Balance Error</b>\n{PAIR} {TIMEFRAME}\n{e}")
            try:
                joblib.dump(bot_state, state_file)
            except Exception as e:
                print(f"Error saving bot state: {e}")
        except Exception as e:
            print(f"[!!] Error main loop: {e}")
            send_telegram(f"[Bot Error]\n{PAIR} {TIMEFRAME}\n{e}")
        # Jeda sesuai timeframe
        sleep_seconds = 60
        if TIMEFRAME.endswith('m'):
            sleep_seconds = int(TIMEFRAME.replace('m', '')) * 60 // 10
        elif TIMEFRAME.endswith('h'):
            sleep_seconds = int(TIMEFRAME.replace('h', '')) * 3600 // 10
        print(f"Sleeping for {sleep_seconds} seconds...")
        time.sleep(sleep_seconds)
    print("=== 1x Trading Selesai, Bot Berhenti ===")
else:
    print("--- Simulasi selesai. Aktifkan IS_LIVE untuk menjalankan trading nyata. ---")