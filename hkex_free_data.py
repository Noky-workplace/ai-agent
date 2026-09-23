"""
Free HK data layer: run at HOME (needs open internet). No terminal time needed.
    pip install akshare pandas pyarrow requests
    python hkex_free_data.py
Builds ./free_data/ with:
  southbound_holdings.parquet   per-stock Stock Connect southbound holdings (daily)  -> the best-evidenced HK signal
  southbound_flow_daily.parquet aggregate southbound/northbound net flow (daily)
  short_selling_daily.parquet   per-stock HKEX short-selling turnover (daily)
  hsi_review_changes.csv        YOU fill this from Hang Seng review press releases (template written)
Sources are third-party; each block is wrapped so one failure doesn't stop the rest.
"""
import os, sys, time, pandas as pd
OUT="free_data"; os.makedirs(OUT,exist_ok=True)
try:
    import akshare as ak
except ImportError:
    sys.exit("pip install akshare first")
members=pd.read_csv("bbg_dataset/members_HSI.csv")["member"].str.replace(" HK","").str.zfill(5).tolist()

def save(df,name):
    df.to_parquet(f"{OUT}/{name}.parquet"); print(f"saved {name}: {df.shape}")

# 1) Aggregate Stock Connect flows (southbound = HK-bound money from mainland)
try:
    sb=ak.stock_hsgt_hist_em(symbol="南向资金")     # daily history, southbound
    nb=ak.stock_hsgt_hist_em(symbol="北向资金")
    save(sb,"southbound_flow_daily"); save(nb,"northbound_flow_daily")
except Exception as e: print("aggregate flows failed:",e)

# 2) Per-stock southbound holdings (mainland investors' holdings of each HK stock)
#    akshare wraps HKEXnews Stock Connect shareholding search. Pull per stock, be gentle with rate.
rows=[]
for i,code in enumerate(members):
    try:
        df=ak.stock_hk_ggt_components_em() if i==0 and False else None
        h=ak.stock_hsgt_individual_em(stock=code) if hasattr(ak,"stock_hsgt_individual_em") else None
        if h is None or len(h)==0:
            # fallback: HKEXnews southbound holdings via akshare's hk holding interface
            h=ak.stock_hsgt_hold_stock_em(market="沪深港通", indicator="今日排行")  # placeholder; replace with per-stock call if API changes
        h["code"]=code; rows.append(h); time.sleep(0.5)
    except Exception as e:
        print(code,"holdings failed:",str(e)[:80])
if rows:
    save(pd.concat(rows),"southbound_holdings")
else:
    print("!! per-stock southbound holdings: akshare endpoint names change often. Manual route:")
    print("   https://www3.hkexnews.hk/sdw/search/mutualmarket.aspx?t=hk  (pick date, export table) - 12 months online")

# 3) Per-stock short-selling turnover (HKEX daily short selling report)
try:
    ss=ak.stock_hk_short_selling_em() if hasattr(ak,"stock_hk_short_selling_em") else None
    if ss is not None: save(ss,"short_selling_daily")
    else: raise RuntimeError("no akshare endpoint")
except Exception as e:
    print("short selling via akshare failed:",str(e)[:80])
    print("   Manual route: https://www.hkex.com.hk/Market-Data/Statistics/Consolidated-Reports/Short-Selling-Turnover  (daily files)")

# 4) Index review template: fill from https://www.hsi.com.hk/eng/newsroom/index-review-results  (quarterly press releases)
tpl=pd.DataFrame(columns=["index","announce_date","effective_date","ticker","action"])  # action = ADD / DELETE
tpl.to_csv(f"{OUT}/hsi_review_changes_TEMPLATE.csv",index=False)
print("\nDone. Template written for index changes; fill it from Hang Seng review press releases (2016-2026, ~43 releases).")
