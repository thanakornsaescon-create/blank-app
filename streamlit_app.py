# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re, warnings, logging
warnings.filterwarnings("ignore")
for name in ('prophet','cmdstanpy','pystan'):
    logging.getLogger(name).setLevel(logging.CRITICAL)

# ========== Optional libs (safe import) ==========
_HAS_PM = False
try:
    from pmdarima import auto_arima
    _HAS_PM = True
except Exception:
    _HAS_PM = False

_HAS_PROPHET = True
try:
    from prophet import Prophet
except Exception:
    _HAS_PROPHET = False

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ========== Config ==========
ZERO_THR = 0.49
MIN_ACTIVE_DAYS = 2
MAX_GAP_MONTHS  = 1
SERVICE_LEVELS  = {90:1.2816, 95:1.6449, 99:2.3263}

# ========== Helpers ==========
def load_aes_excel(file):
    # รองรับ Excel/CSV
    name = file.name.lower()
    if name.endswith((".xls",".xlsx",".xlsm")):
        raw = pd.read_excel(file, header=None)
        REQ_COLS = {'สถานะเอกสาร','วันที่เอกสาร','รหัสสินค้า','ชื่อสินค้า','หน่วยนับ','จำนวนจ่ายสินค้า'}
        header_row = None
        for i, row in raw.iterrows():
            vals = set(row.astype(str).str.strip())
            if REQ_COLS.issubset(vals):
                header_row = i; break
        if header_row is None:
            st.error("ไม่พบแถวหัวตารางที่มีคอลัมน์ที่ต้องการ"); st.stop()
        cols = raw.iloc[header_row].astype(str).str.strip().tolist()
        df = raw.iloc[header_row+1:].copy()
        df.columns = cols
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df = df.dropna(how='all')
        return df
    else:
        return pd.read_csv(file)

def _detect_dayfirst(series, sample_size=400):
    s = pd.Series(series).dropna().astype(str).head(sample_size)
    def _try(dayfirst_flag):
        return pd.to_datetime(s, errors='coerce', dayfirst=dayfirst_flag, infer_datetime_format=True).notna().sum()
    return True if _try(True) > _try(False) else False

def robust_parse_datetime(x, dayfirst_flag=True):
    if pd.isna(x): return pd.NaT
    if isinstance(x, (int, float, np.integer, np.floating)):
        try: return pd.to_datetime(x, origin='1899-12-30', unit='D')
        except: return pd.NaT
    txt = str(x).strip()
    dt = pd.to_datetime(txt, errors='coerce', dayfirst=dayfirst_flag, infer_datetime_format=True)
    if pd.notna(dt): return dt
    m = re.search(r'(\d{1,2})\D+(\d{1,2})\D+(\d{4})', txt)
    if m:
        a,b,y = m.groups()
        try:
            if dayfirst_flag:
                return pd.to_datetime(f'{a}-{b}-{y}', dayfirst=True, format='%d-%m-%Y')
            else:
                return pd.to_datetime(f'{a}-{b}-{y}', dayfirst=False, format='%m-%d-%Y')
        except: return pd.NaT
    return pd.NaT

def _fit_arima_fallback(y, steps):
    y = pd.Series(y).astype(float)
    best_aic = np.inf; best_res = None
    for d in range(0, 3):
        for p in range(0, 4):
            for q in range(0, 4):
                if p==q==d==0: continue
                try:
                    res = SARIMAX(y, order=(p,d,q), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    if res.aic < best_aic: best_aic, best_res = res.aic, res
                except: continue
    if best_res is None:
        fit_in_sample = y.rolling(3, min_periods=1).mean()
        fut = [float(fit_in_sample.iloc[-1])] * steps
        return fit_in_sample, fut
    pred = best_res.get_prediction(start=0, end=len(y)-1, dynamic=False)
    fit = pd.Series(pred.predicted_mean, index=y.index).astype(float).fillna(method='bfill').fillna(method='ffill')
    fut = list(best_res.get_forecast(steps=steps).predicted_mean.astype(float))
    return fit, fut

def _infer_prophet_freq(ds: pd.Series, fallback="D"):
    ds = pd.to_datetime(ds).sort_values()
    if len(ds) < 3: return fallback
    diffs = ds.diff().dropna()
    if (diffs.dt.days == 7).mean() > 0.7: return "W"
    if pd.Series(ds).dt.is_month_start.mean() > 0.7: return "MS"
    try:
        pf = pd.infer_freq(ds)
        if pf:
            if pf.startswith("W-"): return "W"
            if pf in {"MS","M"}:   return "MS"
            if pf == "D":         return "D"
    except: pass
    return fallback

def choose_last_tx_real(g, min_active_days=MIN_ACTIVE_DAYS, max_gap_months=MAX_GAP_MONTHS):
    gg = g.copy()
    gg['y'] = pd.to_numeric(gg['y'], errors='coerce').fillna(0.0)
    gg = gg.dropna(subset=['ds'])
    nz = gg.loc[np.abs(gg['y']) > 0, ['ds','y']].copy()
    if nz.empty:
        return pd.to_datetime(gg['ds'].max())
    nz['mon'] = nz['ds'].dt.to_period('M')
    day_counts = (nz.assign(day=nz['ds'].dt.normalize())
                  .groupby('mon')['day'].nunique().sort_index())
    good_mons = day_counts[day_counts >= min_active_days].index.to_list()
    if not good_mons:
        return nz['ds'].max()
    good_mons.sort()
    i = len(good_mons) - 1
    while i > 0 and (good_mons[i].ordinal - good_mons[i-1].ordinal) > max_gap_months:
        i -= 1
    chosen_mon = good_mons[i]
    return nz.loc[nz['mon'].eq(chosen_mon), 'ds'].max()

def _resolve_lt_for(code, lt_map_hist, lt_ui=7):
    if isinstance(lt_map_hist, dict):
        v = lt_map_hist.get(code)
        if v is not None and v > 0:
            return int(v)
    return int(lt_ui)

# ========== Leadtime (optional file) ==========
TH_MONTHS = {'ม.ค':1,'ก.พ':2,'มี.ค':3,'เม.ย':4,'พ.ค':5,'มิ.ย':6,'ก.ค':7,'ส.ค':8,'ก.ย':9,'ต.ค':10,'พ.ย':11,'ธ.ค':12}
def _parse_thai_date(x):
    if pd.isna(x): return pd.NaT
    if isinstance(x,(int,float,np.integer,np.floating)):
        try: return pd.to_datetime(x, origin='1899-12-30', unit='D')
        except: return pd.NaT
    s = str(x).strip()
    for th,m in TH_MONTHS.items(): s = re.sub(rf'({th}\.?)', f'{m:02d}', s)
    dt = pd.to_datetime(s, errors='coerce', dayfirst=True, infer_datetime_format=True)
    if pd.isna(dt): return pd.NaT
    yr = re.search(r'(\d{4})', s)
    if yr and int(yr.group(1)) >= 2400: dt = dt - pd.DateOffset(years=543)
    return dt

def build_lt_map(file):
    if file is None: return {}
    ext = file.name.lower()
    dfh = pd.read_excel(file) if ext.endswith(('.xls','.xlsx','.xlsm')) else pd.read_csv(file)

    # pick columns
    def _pick(cols, cands):
        for c in cands:
            if c in cols: return c
        return None

    C_ITEM = ['รหัสสินค้า','ItemCode','SKU','Material','รหัส สินค้า','รหัสสินค้า ']
    C_LTC  = ['ระยะเวลาส่งมอบ','ระยะเวลารอคอย','Lead Time','LeadTime','LT(วัน)']
    C_PO   = ['วันที่เอกสาร PO','วันที่ใบสั่งซื้อ','PO Date','PO_Date']
    C_RG   = ['วันที่ RG','วันที่รับเข้า','GR Date','Receive_Date']
    C_ST_PO= ['สถานะ PO']; C_ST_RG=['สถานะ RG']

    item_col = _pick(dfh.columns, C_ITEM)
    lt_col   = _pick(dfh.columns, C_LTC)
    po_col   = _pick(dfh.columns, C_PO)
    rg_col   = _pick(dfh.columns, C_RG)
    st_po    = _pick(dfh.columns, C_ST_PO)
    st_rg    = _pick(dfh.columns, C_ST_RG)
    if not item_col:
        return {}

    # filter cancel + B/O
    if st_po: dfh = dfh[~dfh[st_po].astype(str).str.replace(" ","").str.contains('ยกเลิก', na=False)]
    if st_rg: dfh = dfh[~dfh[st_rg].astype(str).str.replace(" ","").str.contains('ยกเลิก', na=False)]
    dfh[item_col] = dfh[item_col].astype(str).str.strip().str.upper()
    dfh = dfh[dfh[item_col].str[0].isin(['B','O'])]

    LT_MIN, LT_MAX = 1, 180
    USE_ABS_LT     = True

    # 1) direct LT col
    if lt_col:
        tmp = dfh[[item_col, lt_col]].copy()
        tmp[lt_col] = pd.to_numeric(tmp[lt_col].astype(str).str.replace(',',''), errors='coerce')
        if USE_ABS_LT: tmp[lt_col] = tmp[lt_col].abs()
        tmp = tmp[tmp[lt_col].between(LT_MIN, LT_MAX)]
        if not tmp.empty:
            return tmp.groupby(item_col)[lt_col].mean().round().astype(int).to_dict()

    # 2) fallback: RG - PO
    if po_col and rg_col:
        tmp = dfh[[item_col, po_col, rg_col]].copy()
        tmp['po_d'] = tmp[po_col].apply(_parse_thai_date)
        tmp['rg_d'] = tmp[rg_col].apply(_parse_thai_date)
        tmp = tmp.dropna(subset=['po_d','rg_d'])
        raw_days = (tmp['rg_d'] - tmp['po_d']).dt.days
        tmp['lead_days'] = raw_days.abs() if USE_ABS_LT else raw_days
        tmp = tmp[tmp['lead_days'].between(LT_MIN, LT_MAX)]
        if not tmp.empty:
            return tmp.groupby(item_col)['lead_days'].mean().round().astype(int).to_dict()
    return {}

# ========== Core forecasting ==========
def run_forecast(df_input, title="", future_periods=4, x_end_hard=None, ens_p=0.90,
                 lead_time_days=7, review_days=7, period_kind="W"):
    df_input = df_input[['ds','y']].copy().dropna()
    df_input['ds'] = pd.to_datetime(df_input['ds'])
    df_input = df_input.sort_values('ds')
    data = pd.Series(df_input['y'].values, index=df_input['ds'])
    forecast_df = pd.DataFrame({'y': data}).copy()
    results, forecast_values = {}, {}

    def _eval(actual: pd.Series, pred: pd.Series):
        both = pd.concat([actual, pred], axis=1, keys=['a','p']).dropna()
        if len(both) <= 1: return (np.nan,)*5
        both = both.iloc[1:]
        mask = (both['a'] != 0)
        a, p = both.loc[mask,'a'], both.loc[mask,'p']
        ae = (a - p).abs()
        mae = ae.mean(); mse=((a-p)**2).mean(); rmse=np.sqrt(mse)
        mape=(ae/a.abs()).mean()*100
        smape=(200*ae/(a.abs()+p.abs())).mean()
        return mae,mse,rmse,mape,smape

    # 1) ES
    best_mae = float('inf'); best_alpha=None; best_es=None
    for a in np.linspace(0.01,0.99,99):
        tmp=[data.iloc[0]]
        for i in range(1,len(data)): tmp.append(a*data.iloc[i-1] + (1-a)*tmp[-1])
        es = pd.Series(tmp, index=data.index).clip(lower=0)
        mae,mse,rmse,mape,smape = _eval(data, es)
        if not np.isnan(mae) and mae < best_mae:
            best_mae, best_alpha, best_es = mae, a, es
            best_mse, best_rmse, best_mape, best_smape = mse, rmse, mape, smape
    if best_es is None:
        best_es = pd.Series([data.iloc[-1]]*len(data), index=data.index)
        best_alpha=np.nan; best_mae=best_mse=best_rmse=best_mape=best_smape=np.nan
    forecast_df['ES']=best_es
    results['ES']={'MAE':best_mae,'MSE':best_mse,'RMSE':best_rmse,'MAPE':best_mape,'SMAPE':best_smape,'alpha':best_alpha}
    forecast_values['ES']=[float(best_es.iloc[-1])]*future_periods

    # 2) Holt-Winters
    try:
        hw = ExponentialSmoothing(data, trend='add', seasonal=None, initialization_method="estimated").fit()
        hw_fit = pd.Series(hw.fittedvalues, index=data.index).clip(lower=0)
        mae,mse,rmse,mape,smape = _eval(data, hw_fit)
        forecast_df['Holt-Winters']=hw_fit
        results['Holt-Winters']={'MAE':mae,'MSE':mse,'RMSE':rmse,'MAPE':mape,'SMAPE':smape,'alpha':np.nan}
        forecast_values['Holt-Winters']=list(np.clip(hw.forecast(future_periods),0,None))
    except Exception:
        results['Holt-Winters']={'MAE':np.nan,'MSE':np.nan,'RMSE':np.nan,'MAPE':np.nan,'SMAPE':np.nan,'alpha':np.nan}

    # 3) ARIMA
    try:
        if _HAS_PM:
            arima = auto_arima(data.values, suppress_warnings=True, error_action='ignore', stepwise=True)
            arima_in = pd.Series(arima.predict_in_sample(), index=data.index)
            fut = list(arima.predict(n_periods=future_periods))
        else:
            arima_in, fut = _fit_arima_fallback(data.values, steps=future_periods)
            arima_in.index = data.index
        arima_in = arima_in.clip(lower=0)
        fut = [float(max(0.0,x)) for x in fut]
        mae,mse,rmse,mape,smape = _eval(data, arima_in)
        forecast_df['ARIMA']=arima_in
        results['ARIMA']={'MAE':mae,'MSE':mse,'RMSE':rmse,'MAPE':mape,'SMAPE':smape,'alpha':np.nan}
        forecast_values['ARIMA']=fut
    except Exception:
        results['ARIMA']={'MAE':np.nan,'MSE':np.nan,'RMSE':np.nan,'MAPE':np.nan,'SMAPE':np.nan,'alpha':np.nan}

    # 4) Prophet (ถ้ามี)
    if _HAS_PROPHET:
        try:
            p_df = df_input[['ds','y']].copy()
            p_df['ds']=pd.to_datetime(p_df['ds'])
            freq_tag = _infer_prophet_freq(p_df['ds'])
            m=Prophet(); m.fit(p_df)
            future = m.make_future_dataframe(periods=future_periods, freq=freq_tag)
            fc = m.predict(future)
            prophet_fit = fc.set_index('ds').reindex(data.index)['yhat'].clip(lower=0)
            mae,mse,rmse,mape,smape = _eval(data, prophet_fit)
            forecast_df['Prophet']=prophet_fit
            future_dates = fc['ds'].iloc[-future_periods:]
            forecast_values['Prophet']=[float(max(0.0,v)) for v in fc.set_index('ds').loc[future_dates,'yhat']]
            results['Prophet']={'MAE':mae,'MSE':mse,'RMSE':rmse,'MAPE':mape,'SMAPE':smape,'alpha':np.nan}
        except Exception:
            pass

    # 5) Moving Avg
    best_mae=float('inf'); best_w=None; best_ma=None
    max_w=min(11, max(4, len(data)//4)+1)
    for w in range(3, max_w):
        ma = data.rolling(w, min_periods=w).mean().clip(lower=0)
        mae,mse,rmse,mape,smape = _eval(data, ma)
        if not np.isnan(mae) and mae < best_mae:
            best_mae, best_w, best_ma = mae, w, ma
            best_mse, best_rmse, best_mape, best_smape = mse, rmse, mape, smape
    if best_ma is None:
        best_w=np.nan; best_ma=pd.Series([np.nan]*len(data), index=data.index)
        best_mae=best_mse=best_rmse=best_mape=best_smape=np.nan
    forecast_df['MovingAvg']=best_ma
    results['MovingAvg']={'MAE':best_mae,'MSE':best_mse,'RMSE':best_rmse,'MAPE':best_mape,'SMAPE':best_smape,'alpha':best_w}
    mv_last = best_ma.dropna()
    mv_fill = (mv_last.iloc[-int(best_w):].mean() if len(mv_last) and not np.isnan(best_w) else float(data.iloc[-1]))
    forecast_values['MovingAvg']=[float(max(0.0, mv_fill))]*future_periods

    # Summary tables
    st.write("📊 **Summary** —", title)
    st.dataframe(pd.DataFrame.from_dict(results, orient='index').round(2))

    fut_tbl = pd.DataFrame(forecast_values).T
    fut_tbl.columns = [f"T+{i+1}" for i in range(future_periods)]
    st.write("🔮 **Future Forecasts**")
    st.dataframe(fut_tbl.round(2))

    # Ensemble
    models = list(forecast_values.keys())
    ens_table = None
    if models:
        fut_matrix = np.array([forecast_values[m] for m in models if forecast_values[m] is not None], dtype=float)
        if fut_matrix.size:
            lo = np.quantile(fut_matrix, 1-ens_p, axis=0)
            hi = np.quantile(fut_matrix, ens_p,   axis=0)
            valid_mae = {m: results[m]['MAE'] for m in models if (m in results and pd.notna(results[m]['MAE']) and results[m]['MAE']>0)}
            if valid_mae:
                w = np.array([1.0/valid_mae[m] if m in valid_mae else 0.0 for m in models])
                w = w / (w.sum() if w.sum()>0 else 1.0)
                mu = (w[:,None] * fut_matrix).sum(axis=0)
            else:
                mu = fut_matrix.mean(axis=0)
            ens_table = pd.DataFrame({
                f'Ensemble{int(ens_p*100)}_Low':  lo,
                'Ensemble_WAvg':                 mu,
                f'Ensemble{int(ens_p*100)}_High': hi
            }, index=[f'T+{i+1}' for i in range(future_periods)])
            st.write(f"🧩 **Ensemble range ({int(ens_p*100)}%)**")
            st.dataframe(ens_table.round(2))

    # Inventory Planning (คร่าว ๆ)
    period_days = 7 if period_kind == "W" else 30
    cand = [(k,v) for k,v in results.items() if pd.notna(v['MAE'])]
    best_model = min(cand, key=lambda x: x[1]['MAE'])[0] if cand else 'ES'
    avg_future = float(np.mean(forecast_values.get(best_model, forecast_values['ES'])))
    mean_per_day = avg_future / period_days
    std_period = float(forecast_df['y'].tail(min(30,len(forecast_df))).std(skipna=True))
    std_per_day = std_period / max(np.sqrt(period_days), 1.0)
    z = SERVICE_LEVELS.get(int(ens_p*100), 1.6449)
    lt_days = max(1, int(lead_time_days))
    rp_days = max(1, int(review_days))
    safety_stock = z * std_per_day * np.sqrt(lt_days)
    reorder_point = mean_per_day * lt_days + safety_stock
    max_stock = mean_per_day * (lt_days + rp_days) + safety_stock

    st.write("📦 **Inventory Planning**")
    st.write(f"Service Level ~ {int(ens_p*100)}% (z≈{z:.2f}) | LT={lt_days}d | Review={rp_days}d | Period={period_kind}")
    st.write(f"Mean/day: {mean_per_day:.2f} | Std/day: {std_per_day:.2f}")
    st.write(f"Safety Stock: **{safety_stock:.2f}** | Reorder Point: **{reorder_point:.2f}** | Max Stock: **{max_stock:.2f}**")

    # Plots
    x_end = pd.to_datetime(x_end_hard) if x_end_hard is not None else forecast_df.index.max()
    for model in [c for c in forecast_df.columns if c != 'y']:
        fig, ax = plt.subplots(figsize=(9,3.6))
        ax.plot(forecast_df.index, forecast_df['y'], label='Actual', color='black')
        ax.plot(forecast_df.index, forecast_df[model], label=model)
        ax.set_xlim(forecast_df.index.min(), x_end)
        ax.grid(True); ax.legend(); ax.set_title(f"{model} (in-sample) — {title}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    if ens_table is not None:
        # future index
        if len(forecast_df.index) >= 2:
            step = (forecast_df.index[-1] - forecast_df.index[-2]).days
            if abs(step-7)<=1: freq='7D'
            elif 25<=step<=35: freq='MS' if period_kind=="M" else '30D'
            else: freq=f'{max(step,1)}D'
        else:
            freq='7D' if period_kind=="W" else 'MS'
        last_idx = forecast_df.index.max()
        future_idx = pd.date_range(start=last_idx + pd.to_timedelta(1,'D'), periods=len(ens_table), freq=freq)

        fig, ax = plt.subplots(figsize=(9,3.6))
        ax.plot(forecast_df.index, forecast_df['y'], label='Actual', color='black')
        low = ens_table.iloc[:,0].values; mu=ens_table.iloc[:,1].values; high=ens_table.iloc[:,2].values
        ax.fill_between(future_idx, low, high, alpha=0.2, label=f'Ensemble {int(ens_p*100)}% range')
        ax.plot(future_idx, mu, label='Ensemble weighted avg')
        ax.axvline(x=last_idx, linestyle='--')
        ax.grid(True); ax.legend(); ax.set_title(f"Ensemble Forecast Range — {title}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ========== UI ==========
st.set_page_config(page_title="Per-Item Forecast & Inventory", layout="wide")
st.title("📈 Per-Item Forecast & Inventory (Streamlit)")

with st.sidebar:
    st.header("1) อัปโหลดไฟล์")
    main_file = st.file_uploader("ไฟล์หลัก (Excel/CSV) ที่มี: สถานะเอกสาร / วันที่เอกสาร / รหัสสินค้า / ชื่อสินค้า / หน่วยนับ / จำนวนจ่ายสินค้า", type=['xls','xlsx','xlsm','csv'])
    hist_file = st.file_uploader("(ตัวเลือก) ไฟล์ประวัติ Lead Time (Excel/CSV)", type=['xls','xlsx','xlsm','csv'])

    st.header("2) ตัวเลือก Inventory")
    svc = st.selectbox("Service Level", [90,95,99], index=1)  # default 95
    lt_default = st.number_input("LT fallback (days)", min_value=1, value=7, step=1)
    rp_days = st.number_input("Review period (days)", min_value=1, value=7, step=1)

if main_file is None:
    st.info("⬅️ อัปโหลดไฟล์หลักทางแถบด้านซ้ายก่อน")
    st.stop()

# Load + clean
df = load_aes_excel(main_file)
COL_STATUS='สถานะเอกสาร'; COL_CODE='รหัสสินค้า'; COL_NAME='ชื่อสินค้า'
COL_UNIT='หน่วยนับ'; COL_QTY='จำนวนจ่ายสินค้า'; COL_DATE='วันที่เอกสาร'

if COL_STATUS in df.columns:
    status_clean = df[COL_STATUS].astype(str).str.replace(r'\s+','', regex=True).str.strip()
    df = df[status_clean!='ยกเลิกเอกสาร'].copy()

code_clean = df[COL_CODE].astype(str).str.strip().str.upper()
df = df[(code_clean.str.len()>0) & (code_clean.str[0].isin(['B','O']))].copy()

need_cols = {COL_DATE,COL_CODE,COL_NAME,COL_UNIT,COL_QTY}
missing = need_cols - set(df.columns)
if missing: st.error(f"ไฟล์ขาดคอลัมน์: {missing}"); st.stop()

DAYFIRST = _detect_dayfirst(df[COL_DATE])
df_items = df[list(need_cols)].copy()
df_items['ds'] = df_items[COL_DATE].apply(lambda x: robust_parse_datetime(x, DAYFIRST))
df_items = df_items.dropna(subset=['ds'])
df_items[COL_QTY] = pd.to_numeric(df_items[COL_QTY], errors='coerce').fillna(0.0)

# Build LT map (optional)
lt_map_hist = build_lt_map(hist_file)

# Sidebar filters
with st.sidebar:
    st.header("3) ค้นหา/เลือกสินค้า")
    code_kw = st.text_input("ค้นหารหัส", "")
    name_kw = st.text_input("ค้นหาชื่อ", "")
    # รายการ item
    dfm = (df_items[[COL_CODE,COL_NAME]].drop_duplicates()
           .sort_values([COL_CODE,COL_NAME]).reset_index(drop=True))
    if code_kw: dfm = dfm[dfm[COL_CODE].astype(str).str.contains(code_kw, case=False, na=False)]
    if name_kw: dfm = dfm[dfm[COL_NAME].astype(str).str.contains(name_kw, case=False, na=False)]
    options = [f"[{r[COL_CODE]}] {r[COL_NAME]}" for _,r in dfm.iterrows()]
    sel_items = st.multiselect("เลือกสินค้า (เลือกหลายรายการได้)", options, max_selections=6)
    st.caption(f"พบ {len(options)} รายการ")

    st.header("4) ช่วงเวลา + Horizon")
    ds_min = df_items['ds'].min().date()
    ds_max = df_items['ds'].max().date()
    date_range = st.date_input("ช่วงวันที่", value=(ds_min, ds_max), min_value=ds_min, max_value=ds_max)
    fut_w = st.number_input("T+สัปดาห์", min_value=1, value=4, step=1)
    fut_m = st.number_input("T+เดือน",   min_value=1, value=3, step=1)

if not sel_items:
    st.info("⬅️ เลือกสินค้าอย่างน้อย 1 รายการ แล้วเลื่อนลงมาดูผลลัพธ์")
    st.stop()

start_val, end_val = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
ens_p = SERVICE_LEVELS[svc] / max(2.3263, 2.3263)  # แค่ทำให้ map เป็น 0.90/0.95/0.99 ง่าย ๆ ไม่ต้องแก้โค้ดเดิม
ens_lookup = {90:0.90,95:0.95,99:0.99}
ens_p = ens_lookup[svc]

# Run per item
for label in sel_items:
    code = label.split(']')[0].strip('[')
    name = label.split('] ')[1]

    g = (df_items[(df_items[COL_CODE]==code) & (df_items[COL_NAME]==name)]
         .loc[:, ['ds', COL_QTY]].rename(columns={COL_QTY:'y'}).sort_values('ds'))
    g = g[(g['ds']>=start_val) & (g['ds']<end_val + pd.Timedelta(days=1))]
    if g.empty:
        st.warning(f"- ข้าม [{code}] {name} (ไม่พบข้อมูลในช่วง {start_val.date()} → {end_val.date()})")
        continue

    g['y'] = pd.to_numeric(g['y'], errors='coerce').fillna(0.0)
    last_tx_real = choose_last_tx_real(g)

    # Weekly
    df_week = g.copy(); df_week['ds'] = df_week['ds'].dt.to_period('W').dt.start_time
    df_week = df_week.groupby('ds', as_index=False)['y'].sum()
    df_week['y'] = pd.to_numeric(df_week['y'], errors='coerce').fillna(0.0)
    df_week.loc[df_week['y'].abs() < ZERO_THR, 'y'] = 0.0
    df_week = df_week[df_week['ds'] <= last_tx_real]

    # Monthly
    df_month = g.copy(); df_month['ds'] = df_month['ds'].dt.to_period('M').dt.start_time
    df_month = df_month.groupby('ds', as_index=False)['y'].sum()
    df_month['y'] = pd.to_numeric(df_month['y'], errors='coerce').fillna(0.0)
    df_month.loc[df_month['y'].abs() < ZERO_THR, 'y'] = 0.0
    df_month = df_month[df_month['ds'] <= last_tx_real]

    lt_days = _resolve_lt_for(code, lt_map_hist, lt_ui=lt_default)

    st.markdown(f"---\n**ช่วงที่เลือก:** {start_val.date()} → {end_val.date()} | **last tx (มียอดจริง):** {last_tx_real.date()}")
    st.subheader(f"[{code}] {name} — Weekly")
    run_forecast(df_week,  title=f"[{code}] {name} — Weekly",  future_periods=int(fut_w),
                 x_end_hard=last_tx_real, ens_p=ens_p, lead_time_days=lt_days, review_days=rp_days, period_kind="W")

    st.subheader(f"[{code}] {name} — Monthly")
    run_forecast(df_month, title=f"[{code}] {name} — Monthly", future_periods=int(fut_m),
                 x_end_hard=last_tx_real, ens_p=ens_p, lead_time_days=lt_days, review_days=rp_days, period_kind="M")
