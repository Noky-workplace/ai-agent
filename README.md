# HKEX 模擬交易Bot

## 快速開始

### 1. 安裝依賴
```bash
pip install yfinance pandas numpy tabulate
```

### 2. 基本指令
```bash
# 掃描股票 + 執行策略
python trading_bot.py scan

# 查看持倉和損益
python trading_bot.py status

# 查看交易記錄
python trading_bot.py history

# 重置（清空所有記錄，回到1000 HKD）
python trading_bot.py reset
```

---

## 架構說明

```
trading_bot.py
├── CONFIG              — 資金、手續費設定
├── WATCHLIST           — 監控股票清單
├── fetch_ohlcv()       — 抓取Yahoo Finance數據
├── calc_indicators()   — 計算技術指標
│   ├── MA5/10/20
│   ├── RSI(14)
│   ├── MACD
│   ├── 布林帶
│   └── ATR(14)
├── strategy()          ← ★ 在這裡修改你的策略 ★
├── calc_position_size()— 計算買入股數（預設10%資金）
├── execute_buy/sell()  — 模擬交易執行（含手續費）
├── run_scan()          — 主掃描循環
└── Portfolio           — 持倉/現金管理，自動存檔
```

---

## 修改策略

打開 `trading_bot.py`，找到 `strategy()` 函數：

```python
def strategy(df: pd.DataFrame) -> str:
    # 返回 "BUY" / "SELL" / "HOLD"
    
    last = df.iloc[-1]
    
    # 例子：RSI超賣買入
    if last["RSI"] < 30:
        return "BUY"
    elif last["RSI"] > 70:
        return "SELL"
    
    return "HOLD"
```

可用的指標：`MA5`, `MA10`, `MA20`, `RSI`, `MACD`, `MACD_signal`, `MACD_hist`, `BB_upper`, `BB_lower`, `ATR`

---

## 設定自動掃描（Mac/Linux）

每30分鐘自動掃描一次：
```bash
# 打開crontab
crontab -e

# 加入這行（每30分鐘）
*/30 9-16 * * 1-5 cd /path/to/bot && python trading_bot.py scan >> bot.log 2>&1
```

---

## 里程碑追蹤

| 階段 | 目標     | 回報    |
|------|----------|---------|
| 1    | 1,000 HKD| 起點    |
| 2    | 5,000 HKD| +400%   |
| 3    | 10,000 HKD| +900%  |
| 4    | 100,000 HKD| +9900% |

---

## 免責聲明
這是**模擬工具**，不涉及真實資金。
股市有風險，任何策略都可能虧損。
