import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io
# バージョン互換性対応のため追加
try:
    from packaging.version import parse as parse_version
except ImportError:
    parse_version = None

# --- 1. 設定と定数 ---

# ページ設定（初めに呼び出す必要がある）
try:
    st.set_page_config(page_title="有価証券パフォーマンス分析ダッシュボード", layout="wide")
except st.errors.StreamlitAPIException:
    pass

# カラー設定
COLOR_PRIMARY = "#003366"    # 濃紺
COLOR_PORTFOLIO = COLOR_PRIMARY
COLOR_BENCHMARK = "#A9A9A9"  # ダークグレー
COLOR_TOTAL = "#696969"      # ディムグレー

# 超過リターン/銘柄選択効果(SS)/環境評価の色
COLOR_POSITIVE = "#4169E1"   # プラス（ロイヤルブルー）
COLOR_NEGATIVE = "#DC143C"   # マイナス（クリムゾンレッド）

# 資産配分効果(AA)の色
COLOR_AA_POSITIVE = "#2E8B57" # SeaGreen（資産配分効果+）
COLOR_AA_NEGATIVE = "#FF8C00" # DarkOrange（資産配分効果-）

# 戦略評価（行動）の色
COLOR_STRATEGY_OVER = "#2E8B57"  # SeaGreen（オーバーウェイト）
COLOR_STRATEGY_UNDER = "#DAA520" # Goldenrod（アンダーウェイト）

# その他設定
DEFAULT_EXCLUDE_PATTERN = ["政策保有株式", "（参考）"]
FISCAL_START_MONTH = 4

# Pandasバージョン互換性設定
APPLY_KWARGS = {}
if parse_version:
    try:
        PANDAS_VERSION = parse_version(pd.__version__)
        PANDAS_2_2_0 = parse_version("2.2.0")
        if PANDAS_VERSION >= PANDAS_2_2_0:
            APPLY_KWARGS['include_groups'] = False
    except Exception:
        pass


# --- 2. ユーティリティ関数とサンプルデータ生成 ---

def convert_dfs_to_excel(data_dict):
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                    df_export = df.copy()
                    for col in df_export.select_dtypes(include=['datetime64[ns, UTC]', 'datetimetz', 'datetime64[ns]']).columns:
                         if hasattr(df_export[col].dt, 'tz_localize'):
                             try:
                                df_export[col] = df_export[col].dt.tz_localize(None)
                             except TypeError:
                                 pass
                    df_export.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        st.error(f"Excel変換中にエラーが発生しました: {e}")
        return None
    output.seek(0)
    return output.getvalue()

def generate_sample_data():
    START_DATE = '2022-03-31'
    END_DATE = '2025-03-31'
    DATES = pd.date_range(start=START_DATE, end=END_DATE, freq='ME')

    ASSET_CONFIG = {
        "国内債券（固定金利・国債中心）": {"base": 50000, "vol": 0.02, "drift_p": 0.01, "drift_b": 0.005, "income_r": 0.9, "cf_drift": 0.02, "bm_name": "NOMURA-BPI総合"},
        "国内債券（変動/ASW）": {"base": 20000, "vol": 0.005, "drift_p": 0.015, "drift_b": 0.01, "income_r": 1.0, "cf_drift": 0.01, "bm_name": "TONA複利"},
        "外国債券（為替ヘッジ有・固定）": {"base": 30000, "vol": 0.04, "drift_p": 0.03, "drift_b": 0.025, "income_r": 0.8, "cf_drift": 0.05, "bm_name": "FTSE世界国債（除日、円H）"},
        "国内株式（純投資）": {"base": 15000, "vol": 0.18, "drift_p": 0.09, "drift_b": 0.08, "income_r": 0.3, "cf_drift": 0.00, "bm_name": "TOPIX（配当込み）"},
        "不動産（J-REIT含む）": {"base": 5000, "vol": 0.12, "drift_p": 0.06, "drift_b": 0.05, "income_r": 0.6, "cf_drift": 0.01, "bm_name": "東証REIT指数（配当込み）"},
        "マルチアセット・その他投信": {"base": 8000, "vol": 0.10, "drift_p": 0.05, "drift_b": 0.045, "income_r": 0.5, "cf_drift": 0.02, "bm_name": "バランス型指数"},
        "（参考）政策保有株式": {"base": 10000, "vol": 0.18, "drift_p": 0.07, "drift_b": 0.08, "income_r": 0.3, "cf_drift": -0.01, "bm_name": "TOPIX（配当込み）"},
    }

    np.random.seed(42)
    
    portfolio_data = []
    for asset, config in ASSET_CONFIG.items():
        market_value = config["base"]
        book_value = market_value * 0.98
        for i, date in enumerate(DATES):
            monthly_return = np.random.normal(config["drift_p"] / 12, config["vol"] / np.sqrt(12))
            monthly_cf = 0
            if i > 0:
                monthly_cf = market_value * np.random.normal(config["cf_drift"] / 12, 0.01)

            market_value += monthly_cf
            book_value += monthly_cf 
            market_value_start = market_value
            market_value = market_value * (1 + monthly_return)
            
            income_base = max(config["drift_p"], 0.001) 
            monthly_income = market_value_start * (income_base * config["income_r"] / 12)

            realized_capital_gain = 0
            if i > 0 and np.random.rand() < 0.05:
                sale_book_value = book_value * 0.1
                sale_market_value = sale_book_value * (1 + monthly_return * 5)
                realized_capital_gain = sale_market_value - sale_book_value
                market_value -= sale_market_value
                book_value -= sale_book_value

            if i == 0:
                monthly_income = 0
                realized_capital_gain = 0
            
            portfolio_data.append({
                "年月": date, "資産クラス": asset, "簿価": round(book_value, 2), "時価": round(market_value, 2),
                "インカム収益額": round(monthly_income, 2),
                "キャピタル収益額（実現）": round(realized_capital_gain, 2),
            })
    df_portfolio = pd.DataFrame(portfolio_data)

    benchmark_data = []
    unique_benchmarks = {}
    for config in ASSET_CONFIG.values():
        name = config["bm_name"]
        if name not in unique_benchmarks:
            unique_benchmarks[name] = {"drift_b": config["drift_b"], "vol": config["vol"]}
    for name, config in unique_benchmarks.items():
        for i, date in enumerate(DATES):
            if i == 0: continue
            bm_return = np.random.normal(config["drift_b"] / 12, config["vol"] / np.sqrt(12))
            benchmark_data.append({"年月": date, "ベンチマーク名": name, "月次リターン (%)": round(bm_return * 100, 4)})
    df_benchmark = pd.DataFrame(benchmark_data)

    policy_mix_data = []
    performance_assets = [a for a in ASSET_CONFIG.keys() if "（参考）" not in a]
    total_base_value = sum(ASSET_CONFIG[a]["base"] for a in performance_assets)
    for asset, config in ASSET_CONFIG.items():
        if asset in performance_assets:
            weight = config["base"] / total_base_value
            policy_weight = round(weight * 100, 2)
        else:
            policy_weight = np.nan
        policy_mix_data.append({"資産クラス": asset, "基準配分比率 (%)": policy_weight, "対応ベンチマーク名": config["bm_name"]})
    df_policy_mix = pd.DataFrame(policy_mix_data)

    return convert_dfs_to_excel({
        "ポートフォリオ実績": df_portfolio,
        "ベンチマークリターン": df_benchmark,
        "基準資産配分（設定）": df_policy_mix
    })


# --- 3. データロードと前処理関数 ---

@st.cache_data
def load_data(uploaded_file):
    try:
        xls = pd.ExcelFile(uploaded_file)
        
        required_sheets = ["ポートフォリオ実績", "ベンチマークリターン", "基準資産配分（設定）"]
        if not all(sheet in xls.sheet_names for sheet in required_sheets):
            return None, None, None, None, f"必須シートが不足しています。シート名を確認してください: {required_sheets}"

        df_portfolio = pd.read_excel(xls, sheet_name="ポートフォリオ実績")
        df_benchmark = pd.read_excel(xls, sheet_name="ベンチマークリターン")
        df_policy_mix = pd.read_excel(xls, sheet_name="基準資産配分（設定）")
        
        df_portfolio['年月'] = pd.to_datetime(df_portfolio['年月'], errors='coerce')
        df_benchmark['年月'] = pd.to_datetime(df_benchmark['年月'], errors='coerce')
        df_portfolio = df_portfolio.dropna(subset=['年月'])
        df_benchmark = df_benchmark.dropna(subset=['年月'])

        if 'キャピタル収益額（実現）' in df_portfolio.columns:
             df_portfolio = df_portfolio.rename(columns={'キャピタル収益額（実現）': 'キャピタル収益額'})
        elif 'キャピタル収益額' not in df_portfolio.columns:
             return None, None, None, None, "ポートフォリオ実績シートに「キャピタル収益額」または「キャピタル収益額（実現）」列が必要です。"

        for col in ['簿価', '時価', 'インカム収益額', 'キャピタル収益額']:
             df_portfolio[col] = pd.to_numeric(df_portfolio[col], errors='coerce').fillna(0)

        df_benchmark['月次リターン'] = pd.to_numeric(df_benchmark['月次リターン (%)'], errors='coerce') / 100.0
        df_policy_mix['基準配分比率'] = pd.to_numeric(df_policy_mix['基準配分比率 (%)'], errors='coerce') / 100.0
        
        asset_order = df_policy_mix['資産クラス'].dropna().unique().tolist()
            
        return df_portfolio, df_benchmark, df_policy_mix, asset_order, None
    except Exception as e:
        return None, None, None, None, f"データ読み込みエラー: {e}."

def get_fiscal_year(date, start_month):
    if pd.isna(date): return np.nan
    if date.month >= start_month:
        return date.year
    else:
        return date.year - 1

def calculate_monthly_returns(df_portfolio, start_month):
    df = df_portfolio.copy()
    df = df.sort_values(['資産クラス', '年月'])
    
    df['前月時価'] = df.groupby('資産クラス')['時価'].shift(1)
    df['前月簿価'] = df.groupby('資産クラス')['簿価'].shift(1)
    df['インカム収益額'] = df['インカム収益額'].fillna(0)
    df['キャピタル収益額'] = df['キャピタル収益額'].fillna(0)

    df['当月評価損益'] = df['時価'] - df['簿価']
    df['前月評価損益'] = df['前月時価'] - df['前月簿価']
    df['評価損益変動額（未実現）'] = np.where(
        df['前月評価損益'].notna() & df['前月時価'].notna(),
        df['当月評価損益'] - df['前月評価損益'],
        np.nan
    )
    
    df['トータル収益額'] = df['インカム収益額'] + df['キャピタル収益額'] + df['評価損益変動額（未実現）']
    df['逆算キャッシュフロー'] = (df['時価'] - df['前月時価']) - df['トータル収益額']
    df['平均投資元本'] = df['前月時価'] + df['逆算キャッシュフロー'] * 0.5
    
    MIN_DENOMINATOR = 1e-6 
    df['月次リターン（Rp）'] = np.where(
        (df['平均投資元本'].notna()) & (df['平均投資元本'] > MIN_DENOMINATOR),
        df['トータル収益額'] / df['平均投資元本'],
        np.where(
            (df['平均投資元本'].notna()) & (df['トータル収益額'].abs() < MIN_DENOMINATOR),
            0.0,
            np.nan
        )
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['年度'] = df['年月'].apply(lambda x: get_fiscal_year(x, start_month))
    
    return df

# --- 4. 分析計算ロジック ---

def run_analysis(df_monthly, df_benchmark, df_policy_mix, selected_fy, excluded_assets, is_all_period, rebalance_option):
    df_analysis_monthly = prepare_analysis_data(df_monthly, df_benchmark, df_policy_mix, selected_fy, excluded_assets, is_all_period, rebalance_option)
    
    if df_analysis_monthly.empty:
        return (None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()), df_analysis_monthly, "選択された期間において、パフォーマンス評価対象となる有効なデータがありません。"
    
    try:
        summary, df_monthly_ts, df_brinson, df_asset_performance = calculate_fiscal_performance(df_analysis_monthly, is_all_period)
        results = (summary, df_monthly_ts, df_brinson, df_asset_performance)
        return results, df_analysis_monthly, None
    except Exception as e:
        return (None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()), df_analysis_monthly, f"分析処理中に予期せぬエラーが発生しました: {e}"


def prepare_analysis_data(df_monthly, df_benchmark, df_policy_mix, selected_fy, excluded_assets, is_all_period, rebalance_option):
    df_fy = df_monthly.copy() if is_all_period else df_monthly[(df_monthly['年度'] == selected_fy)].copy()
    
    df_fy = df_fy.dropna(subset=['月次リターン（Rp）'])
    df_fy = df_fy.dropna(subset=['前月時価'])
    df_fy = df_fy[df_fy['前月時価'] >= 0]

    if excluded_assets:
        df_fy = df_fy[~df_fy['資産クラス'].isin(excluded_assets)]
    
    df_policy = df_policy_mix.dropna(subset=['基準配分比率']).copy()
    if excluded_assets:
        df_policy = df_policy[~df_policy['資産クラス'].isin(excluded_assets)]
    
    total_policy_weight = df_policy['基準配分比率'].sum()
    if total_policy_weight > 0:
        df_policy['基準配分比率（Wb）'] = df_policy['基準配分比率'] / total_policy_weight
    else:
        return pd.DataFrame() 
    
    policy_map = df_policy.set_index('資産クラス')[['基準配分比率（Wb）', '対応ベンチマーク名']].to_dict()
    df_fy['基準配分比率（Wb）'] = df_fy['資産クラス'].map(policy_map['基準配分比率（Wb）'])
    df_fy['対応ベンチマーク名'] = df_fy['資産クラス'].map(policy_map['対応ベンチマーク名'])
    df_fy = df_fy.dropna(subset=['基準配分比率（Wb）'])

    total_prev_mv = df_fy.groupby('年月')['前月時価'].transform('sum')
    df_fy['実際配分比率（Wp）'] = np.where(
        total_prev_mv > 0,
        df_fy['前月時価'] / total_prev_mv,
        0
    )
    
    df_bm = df_benchmark[['年月', 'ベンチマーク名', '月次リターン']].rename(columns={'月次リターン': 'ベンチマークリターン（Rb）'})
    df_merged = pd.merge(df_fy, df_bm, left_on=['年月', '対応ベンチマーク名'], right_on=['年月', 'ベンチマーク名'], how='left')
    df_merged = df_merged.dropna(subset=['ベンチマークリターン（Rb）'])

    df_merged = df_merged.sort_values(['資産クラス', '年月'])
    df_merged['基準配分比率_期初（Wb_initial）'] = df_merged['基準配分比率（Wb）']
    
    def calc_cumprod(x): return (1 + x).cumprod()

    if rebalance_option == "リバランスなし（Buy & Hold）":
        df_merged['前月BMリターン'] = df_merged.groupby('資産クラス')['ベンチマークリターン（Rb）'].shift(1).fillna(0)
        df_merged['BM累積指数_前月末'] = df_merged.groupby('資産クラス')['前月BMリターン'].transform(calc_cumprod).fillna(1.0)
        df_merged['仮想BM残高'] = df_merged['基準配分比率_期初（Wb_initial）'] * df_merged['BM累積指数_前月末']
        total_virtual = df_merged.groupby('年月')['仮想BM残高'].transform('sum').fillna(0)
        df_merged['基準配分比率（Wb）'] = np.where(total_virtual > 0, df_merged['仮想BM残高'] / total_virtual, 0)

    elif rebalance_option == "カスタム（年次リバランス）":
        df_merged['前月BMリターン'] = df_merged.groupby(['資産クラス', '年度'])['ベンチマークリターン（Rb）'].shift(1).fillna(0)
        df_merged['BM累積指数_前月末'] = df_merged.groupby(['資産クラス', '年度'])['前月BMリターン'].transform(calc_cumprod).fillna(1.0)
        df_merged['仮想BM残高'] = df_merged['基準配分比率_期初（Wb_initial）'] * df_merged['BM累積指数_前月末']
        total_virtual = df_merged.groupby('年月')['仮想BM残高'].transform('sum').fillna(0)
        df_merged['基準配分比率（Wb）'] = np.where(total_virtual > 0, df_merged['仮想BM残高'] / total_virtual, 0)

    return df_merged

def calculate_fiscal_performance(df_analysis_monthly, is_all_period):
    df = df_analysis_monthly.copy()

    df['Rp_weighted'] = (df['実際配分比率（Wp）'].fillna(0) * df['月次リターン（Rp）'].fillna(0))
    df['Rb_weighted_policy'] = (df['基準配分比率（Wb）'].fillna(0) * df['ベンチマークリターン（Rb）'].fillna(0))

    df_monthly_ts = df.groupby('年月').agg(
        Portfolio=('Rp_weighted', 'sum'),
        Benchmark=('Rb_weighted_policy', 'sum')
    ).reset_index()
    
    N_months = df_monthly_ts['年月'].nunique()

    def geometric_mean(returns):
        valid_returns = returns.dropna()
        if valid_returns.empty: return 0.0
        return (1 + valid_returns).prod() - 1

    Rp_period = geometric_mean(df_monthly_ts['Portfolio'])
    Rb_period = geometric_mean(df_monthly_ts['Benchmark'])

    if is_all_period and N_months > 0:
        Rp_fy = (1 + Rp_period) ** (12 / N_months) - 1
        Rb_fy = (1 + Rb_period) ** (12 / N_months) - 1
    else:
        Rp_fy = Rp_period
        Rb_fy = Rb_period

    Excess_Return_fy = Rp_fy - Rb_fy

    def annualized_risk(returns):
        valid_returns = returns.dropna()
        if len(valid_returns) < 2: return np.nan
        return valid_returns.std(ddof=1) * np.sqrt(12)

    Risk_P_fy = annualized_risk(df_monthly_ts['Portfolio'])
    Risk_B_fy = annualized_risk(df_monthly_ts['Benchmark'])

    def calc_asset_returns(x):
        rp_period = geometric_mean(x['月次リターン（Rp）'])
        rb_period = geometric_mean(x['ベンチマークリターン（Rb）'])
        n_months_asset = x['年月'].nunique()
        
        if is_all_period and n_months_asset > 0:
            rp_fy_asset = (1 + rp_period) ** (12 / n_months_asset) - 1
            rb_fy_asset = (1 + rb_period) ** (12 / n_months_asset) - 1
        else:
            rp_fy_asset = rp_period
            rb_fy_asset = rb_period
            
        return pd.Series({'Rp_fy': rp_fy_asset, 'Rb_fy': rb_fy_asset})
    
    df_asset_returns = df.groupby('資産クラス').apply(
        calc_asset_returns, **APPLY_KWARGS
    ).reset_index()

    df_asset_risk = df.groupby('資産クラス').apply(
        lambda x: pd.Series({'Risk_P_fy': annualized_risk(x['月次リターン（Rp）'])}),
        **APPLY_KWARGS
    ).reset_index()
    
    df_asset_performance = pd.merge(df_asset_returns, df_asset_risk, on='資産クラス', how='outer')

    df_weights_fy = df.groupby('資産クラス').agg(
        Wp_avg=('実際配分比率（Wp）', 'mean'),
        Wb_avg=('基準配分比率（Wb）', 'mean')
    ).reset_index()
    
    df_brinson = pd.merge(df_weights_fy, df_asset_performance, on='資産クラス')
    
    df_brinson['AA_行動（Wp-Wb）'] = df_brinson['Wp_avg'] - df_brinson['Wb_avg']
    df_brinson['AA_環境（Rb-R_total）'] = df_brinson['Rb_fy'] - Rb_fy
    df_brinson['資産配分効果'] = df_brinson['AA_行動（Wp-Wb）'] * df_brinson['AA_環境（Rb-R_total）']

    df_brinson['SS_行動（Rp-Rb）'] = df_brinson['Rp_fy'] - df_brinson['Rb_fy']
    df_brinson['銘柄選択効果'] = df_brinson['Wp_avg'] * df_brinson['SS_行動（Rp-Rb）']
    
    df_brinson['トータル効果'] = df_brinson['資産配分効果'] + df_brinson['銘柄選択効果']
    
    AA_total = df_brinson['資産配分効果'].sum()
    SS_total = df_brinson['銘柄選択効果'].sum()
    Total_Effect = AA_total + SS_total
    
    Error = Excess_Return_fy - Total_Effect

    summary = {
        'Rp_fy': Rp_fy, 'Rb_fy': Rb_fy, 'Excess_Return_fy': Excess_Return_fy,
        'Risk_P_fy': Risk_P_fy, 'Risk_B_fy': Risk_B_fy,
        'AA_total': AA_total, 'SS_total': SS_total, 'Error': Error,
    }
    
    return summary, df_monthly_ts, df_brinson, df_asset_performance

# --- 5. 可視化関数と示唆生成 ---

def generate_insights(summary, df_brinson, return_label, analysis_period_label):
    insights = []
    excess_return = summary['Excess_Return_fy']
    Rp = summary['Rp_fy']
    Rb = summary['Rb_fy']

    direction = "上回りました" if excess_return >= 0 else "下回りました"
    perf_summary_text = f"**全体評価:** {analysis_period_label}のポートフォリオはベンチマークを{excess_return:+.2%}{direction}。（実績：{Rp:+.2%}、BM：{Rb:+.2%}）"

    if Rb < -0.01:
         market_context = f"（市場環境が厳しい中での結果）"
         insights.append(f"{perf_summary_text} {market_context}")
    else:
         insights.append(perf_summary_text)

    aa_total = summary['AA_total']
    ss_total = summary['SS_total'] + summary['Error']

    aa_abs = abs(aa_total)
    ss_abs = abs(ss_total)

    if aa_abs == 0 and ss_abs == 0:
        main_driver = "要因は特定できませんでした。"
    elif aa_abs > ss_abs * 1.5: 
        main_driver = "主に**資産配分戦略**（基準配分からの調整）によるもの"
    elif ss_abs > aa_abs * 1.5:
        main_driver = "主に**銘柄選択**によるもの"
    else:
        main_driver = "**資産配分**と**銘柄選択**の両方が影響"
    
    insights.append(f"**主な要因:** 超過リターンは、{main_driver}です。(配分効果: {aa_total:+.2%}, 銘柄選択効果: {ss_total:+.2%})")

    if df_brinson.empty:
        return insights

    df_factors = df_brinson.melt(id_vars=['資産クラス'], value_vars=['資産配分効果', '銘柄選択効果'], var_name='要因種別', value_name='寄与度')
    
    if not df_factors['寄与度'].isnull().all():
        try:
            best_contributor = df_factors.loc[df_factors['寄与度'].idxmax()]
            if best_contributor['寄与度'] > 0.001:
                factor_name = "配分" if best_contributor['要因種別'] == '資産配分効果' else "銘柄選択"
                insights.append(f"🟢 **最大の貢献要因:** **{best_contributor['資産クラス']}**の**{factor_name}**が最もプラスに寄与しました ({best_contributor['寄与度']:+.2%})。")
        except (ValueError, KeyError):
            pass

    if not df_factors['寄与度'].isnull().all():
        try:
            worst_contributor = df_factors.loc[df_factors['寄与度'].idxmin()]
            if worst_contributor['寄与度'] < -0.001:
                factor_name = "配分" if worst_contributor['要因種別'] == '資産配分効果' else "銘柄選択"
                insights.append(f"🔴 **最大の毀損要因:** **{worst_contributor['資産クラス']}**の**{factor_name}**が最もマイナスに寄与しました ({worst_contributor['寄与度']:+.2%})。")
        except (ValueError, KeyError):
            pass

    if 'Wp_avg' in df_brinson.columns and 'Wb_avg' in df_brinson.columns:
        df_positive_allocation = df_brinson[(df_brinson['Wp_avg'] > df_brinson['Wb_avg']) & (df_brinson['資産配分効果'] > 0.001)]
        if not df_positive_allocation.empty:
            df_positive_allocation_clean = df_positive_allocation.drop(columns=['Action_abs', '資産配分効果_abs', '銘柄選択効果_abs'], errors='ignore')
            if not df_positive_allocation_clean.empty:
                if pd.api.types.is_numeric_dtype(df_positive_allocation_clean['資産配分効果']):
                     try:
                         best_strategy = df_positive_allocation_clean.loc[df_positive_allocation_clean['資産配分効果'].idxmax()]
                         insights.append(f"⭐ **戦術的判断の評価:** **{best_strategy['資産クラス']}**へのオーバーウェイト（BM比{best_strategy['Wp_avg']-best_strategy['Wb_avg']:+.1%}）は成功し、リターン向上に貢献しました。")
                     except (ValueError, KeyError):
                         pass

    return insights

def plot_cumulative_returns_with_excess(df_monthly_ts):
    if df_monthly_ts.empty: return alt.Chart(pd.DataFrame()).mark_text(text="データなし").properties(height=400)

    df = df_monthly_ts.copy()
    df['Portfolio_Cum'] = (1 + df['Portfolio']).cumprod() - 1
    df['Benchmark_Cum'] = (1 + df['Benchmark']).cumprod() - 1
    df_melted = df.melt(id_vars=['年月'], value_vars=['Portfolio_Cum', 'Benchmark_Cum'], var_name='凡例', value_name='累積リターン')
    df_melted['凡例'] = df_melted['凡例'].replace({'Portfolio_Cum': 'ポートフォリオ', 'Benchmark_Cum': '複合ベンチマーク'})

    df['月次超過リターン'] = df['Portfolio'] - df['Benchmark']
    df['年月_str'] = df['年月'].dt.strftime('%Y/%m')
    df_melted['年月_str'] = df_melted['年月'].dt.strftime('%Y/%m')
    
    line_min = df_melted['累積リターン'].min()
    line_max = df_melted['累積リターン'].max()
    line_abs_max = max(abs(line_min) if not np.isnan(line_min) else 0, abs(line_max) if not np.isnan(line_max) else 0, 0.01) 
    line_domain = [-line_abs_max * 1.1, line_abs_max * 1.1] 

    bar_min = df['月次超過リターン'].min()
    bar_max = df['月次超過リターン'].max()
    bar_abs_max = max(abs(bar_min) if not np.isnan(bar_min) else 0, abs(bar_max) if not np.isnan(bar_max) else 0, 0.005)
    bar_domain = [-bar_abs_max * 1.1, bar_abs_max * 1.1] 

    base_x_axis = alt.X('年月_str:O', axis=alt.Axis(title='年月', labelAngle=-45, grid=True, labelLimit=0), scale=alt.Scale(paddingOuter=0.1))
    base = alt.Chart(df).encode(x=base_x_axis)

    line = alt.Chart(df_melted).mark_line(size=2.5, point=True).encode(
        x=base_x_axis,
        y=alt.Y('累積リターン:Q', axis=alt.Axis(title='累積リターン', format=".1%", titleColor=COLOR_PORTFOLIO, grid=False, orient='left'), scale=alt.Scale(domain=line_domain)),
        color=alt.Color('凡例:N', scale=alt.Scale(domain=['ポートフォリオ', '複合ベンチマーク'], range=[COLOR_PORTFOLIO, COLOR_BENCHMARK]), legend=alt.Legend(orient='top')),
        strokeDash=alt.condition(alt.datum.凡例 == '複合ベンチマーク', alt.value([5, 5]), alt.value([0])),
        tooltip=[alt.Tooltip('年月_str', title='年月'), alt.Tooltip('凡例:N'), alt.Tooltip('累積リターン:Q', format=".2%")]
    )

    bar = base.mark_bar(opacity=0.5, size=10).encode(
        y=alt.Y('月次超過リターン:Q', axis=alt.Axis(title='月次超過リターン', format=".2%", titleColor=COLOR_NEGATIVE, grid=False, orient='right'), scale=alt.Scale(domain=bar_domain)),
        color=alt.condition(alt.datum.月次超過リターン > 0, alt.value(COLOR_POSITIVE), alt.value(COLOR_NEGATIVE)),
        tooltip=[alt.Tooltip('年月_str', title='年月'), alt.Tooltip('月次超過リターン:Q', format=".2%")]
    )
    
    zero_line_left = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(strokeDash=[2, 2], color='gray', strokeWidth=0.5).encode(y='y:Q')

    final_chart = alt.layer(zero_line_left, line, bar).properties(
        title="累積リターン推移 と 月次超過リターン", height=400, padding={"left": 50, "right": 50}
    ).resolve_scale(y='independent').interactive()
    
    return final_chart

def plot_brinson_waterfall(summary):
    if summary is None: return alt.Chart(pd.DataFrame()).mark_text(text="データなし").properties(height=400)
    
    AA_adjusted = summary['AA_total']
    SS_adjusted = summary['SS_total'] + summary['Error']

    data = {
        "Category": ["ベンチマーク", "資産配分効果", "銘柄選択効果", "ポートフォリオ"],
        "Value": [summary['Rb_fy'], AA_adjusted, SS_adjusted, summary['Rp_fy']],
    }
    df = pd.DataFrame(data)
    
    df['End'] = 0.0
    df['Start'] = 0.0
    current_level = 0.0
    for i in range(len(df)):
        category = df.loc[i, 'Category']
        value = df.loc[i, 'Value']
        if np.isnan(value): value = 0.0
        if category in ["ベンチマーク", "ポートフォリオ"]:
            df.loc[i, 'Start'] = 0.0
            df.loc[i, 'End'] = value
            if category == 'ベンチマーク': current_level = value
        else:
            df.loc[i, 'Start'] = current_level
            current_level += value
            df.loc[i, 'End'] = current_level
    if not df[df['Category'] == 'ポートフォリオ'].empty:
         df.loc[df['Category'] == 'ポートフォリオ', 'End'] = current_level

    def get_color(row):
        if row['Category'] == "ベンチマーク": return COLOR_BENCHMARK
        if row['Category'] == "ポートフォリオ": return COLOR_PORTFOLIO
        if row['Category'] == "資産配分効果": return COLOR_AA_POSITIVE if row['Value'] >= 0 else COLOR_AA_NEGATIVE
        return COLOR_POSITIVE if row['Value'] >= 0 else COLOR_NEGATIVE
    df['Color'] = df.apply(get_color, axis=1)

    y_min = df[['Start', 'End']].min().min()
    y_max = df[['Start', 'End']].max().max()
    y_range = y_max - y_min
    if y_range == 0 or np.isnan(y_range): y_range = 0.1
    threshold = y_range * 0.05
    label_offset = y_range * 0.03

    def calculate_label_props(row):
        is_small = abs(row['Value']) < threshold
        is_positive = row['Value'] >= 0
        if is_small:
            y_pos = row['End'] + (label_offset if is_positive else -label_offset)
            color = 'black'
        else:
            y_pos = (row['Start'] + row['End']) / 2
            color = 'white' 
        return pd.Series([y_pos, color])

    df[['LabelYPosition', 'LabelColor']] = df.apply(calculate_label_props, axis=1)

    def format_value(row):
        if row['Category'] in ["ベンチマーク", "ポートフォリオ"]: return f"{row['Value']:.2%}"
        else: return f"{row['Value']:+.2%}"
    df['ValueFormatted'] = df.apply(format_value, axis=1)

    df['NextCategory'] = df['Category'].shift(-1)
    df_rules = df.dropna(subset=['NextCategory'])

    sort_order = df['Category'].tolist()

    chart = alt.Chart(df).mark_bar(size=70).encode(
        x=alt.X('Category:N', sort=sort_order, axis=alt.Axis(title="要因", labelAngle=0, labelLimit=0)),
        y=alt.Y('Start:Q', axis=alt.Axis(title='リターン', format=".1%")),
        y2='End:Q', color=alt.Color('Color:N', scale=None)
    )
    
    text = alt.Chart(df).mark_text(align='center', baseline='middle', fontWeight='bold', fontSize=12).encode(
        x=alt.X('Category:N', sort=sort_order), y=alt.Y('LabelYPosition:Q'), 
        color=alt.Color('LabelColor:N', scale=None), text=alt.Text('ValueFormatted:N')
    )

    rules = alt.Chart(df_rules).mark_rule(strokeDash=[5, 5], strokeWidth=1, color='gray').encode(
        y='End:Q', x=alt.X('Category:N', sort=sort_order), x2=alt.X2('NextCategory:N')
    )

    final_chart = alt.layer(chart, text, rules).properties(title="パフォーマンス要因分析（ブリンソン分解）", height=400).resolve_scale(y='shared')
    
    return final_chart

# 【修正】直近月の円グラフに数値ラベルを追加
def plot_latest_allocation_pie_charts(df_analysis_monthly, asset_order):
    if df_analysis_monthly.empty: return alt.Chart(pd.DataFrame()).mark_text(text="データなし")
    latest_month = df_analysis_monthly['年月'].max()
    df_latest = df_analysis_monthly[df_analysis_monthly['年月'] == latest_month].copy()
    
    month_str = latest_month.strftime('%Y年%m月')
    color_scale = alt.Scale(scheme='category10', domain=asset_order)
    
    # --- 実績ポートフォリオ ---
    base_wp = alt.Chart(df_latest).encode(
        theta=alt.Theta('実際配分比率（Wp）:Q', stack=True),
        color=alt.Color('資産クラス:N', sort=asset_order, scale=color_scale, legend=alt.Legend(title="資産クラス")),
        tooltip=['資産クラス', alt.Tooltip('実際配分比率（Wp）:Q', format='.2%')]
    )
    pie_wp = base_wp.mark_arc(outerRadius=120, innerRadius=50)
    # 文字重なりを防ぐため、3%以上のスライスのみラベル表示
    text_wp = base_wp.mark_text(radius=85, size=12, color='white').encode(
        text=alt.Text('実際配分比率（Wp）:Q', format='.1%')
    ).transform_filter(alt.datum['実際配分比率（Wp）'] >= 0.03)
    
    chart_wp = alt.layer(pie_wp, text_wp).properties(title=f"実績ポートフォリオ ({month_str})", width=300)
    
    # --- ベンチマーク ---
    base_wb = alt.Chart(df_latest).encode(
        theta=alt.Theta('基準配分比率（Wb）:Q', stack=True),
        color=alt.Color('資産クラス:N', sort=asset_order, scale=color_scale, legend=None),
        tooltip=['資産クラス', alt.Tooltip('基準配分比率（Wb）:Q', format='.2%')]
    )
    pie_wb = base_wb.mark_arc(outerRadius=120, innerRadius=50)
    text_wb = base_wb.mark_text(radius=85, size=12, color='white').encode(
        text=alt.Text('基準配分比率（Wb）:Q', format='.1%')
    ).transform_filter(alt.datum['基準配分比率（Wb）'] >= 0.03)
    
    chart_wb = alt.layer(pie_wb, text_wb).properties(title=f"ベンチマーク ({month_str})", width=300)
    
    return alt.hconcat(chart_wp, chart_wb).resolve_scale(color='shared')

# 【修正】月次の100%積み上げ棒グラフ（棒の幅を広げる）
def plot_allocation_bar_chart(df_analysis_monthly, asset_order):
    if df_analysis_monthly.empty: return alt.Chart(pd.DataFrame()).mark_text(text="データなし").properties(height=300)
    df = df_analysis_monthly.copy()
    
    # X軸を「日付型（T）」から「カテゴリ型（文字列のO）」に変更し、棒同士の隙間を小さくする
    df['年月_str'] = df['年月'].dt.strftime('%Y/%m')
    
    color_scale = alt.Scale(scheme='category10', domain=asset_order)

    # paddingInner=0.05 で隙間を最小限にし、面グラフに近い太い棒グラフにする
    base_x = alt.X('年月_str:O', axis=alt.Axis(title='年月', labelAngle=-45, grid=False, labelLimit=0), scale=alt.Scale(paddingInner=0.05))

    chart_wp = alt.Chart(df).mark_bar(opacity=0.9).encode(
        x=base_x,
        y=alt.Y('実際配分比率（Wp）:Q', stack='normalize', axis=alt.Axis(title='Wp（実績）', format='%')),
        color=alt.Color('資産クラス:N', sort=asset_order, scale=color_scale, legend=alt.Legend(title="資産クラス", orient='right')),
        tooltip=[alt.Tooltip('年月_str:O', title='年月'), '資産クラス', alt.Tooltip('実際配分比率（Wp）:Q', format='.2%', title='実績構成比')]
    ).properties(height=250, title="実績構成比の推移（Wp）")

    chart_wb = alt.Chart(df).mark_bar(opacity=0.9).encode(
        x=base_x,
        y=alt.Y('基準配分比率（Wb）:Q', stack='normalize', axis=alt.Axis(title='Wb（BM）', format='%')),
        color=alt.Color('資産クラス:N', sort=asset_order, scale=color_scale, legend=None), 
        tooltip=[alt.Tooltip('年月_str:O', title='年月'), '資産クラス', alt.Tooltip('基準配分比率（Wb）:Q', format='.2%', title='BM構成比')]
    ).properties(height=250, title="ベンチマーク構成比の推移（Wb）")

    return alt.vconcat(chart_wp, chart_wb).resolve_scale(color='shared').interactive()


# 【修正】資産クラス別 リターン比較に数値ラベルを追加
def plot_asset_returns_comparison(df_asset_performance, return_label, asset_order):
    if df_asset_performance.empty: return alt.Chart(pd.DataFrame()).mark_text(text="データなし").properties(height=550)

    df = df_asset_performance.copy()
    df['資産超過リターン'] = df['Rp_fy'] - df['Rb_fy']
    sort_order = df.sort_values('資産超過リターン', ascending=False)['資産クラス'].tolist()

    df_p = df[['資産クラス', 'Rp_fy']].rename(columns={'Rp_fy': 'リターン'})
    df_p['凡例'] = 'ポートフォリオ'
    df_p['OffsetGroup'] = 'ポートフォリオ'
    df_b = df[['資産クラス', 'Rb_fy']].rename(columns={'Rb_fy': 'リターン'})
    df_b['凡例'] = 'ベンチマーク'
    df_b['OffsetGroup'] = 'ベンチマーク'
    df_excess_plus = df[['資産クラス', '資産超過リターン']].copy()
    df_excess_plus['リターン'] = df_excess_plus['資産超過リターン'].apply(lambda x: x if x >= 0 else np.nan)
    df_excess_plus['凡例'] = '超過リターン(+)'
    df_excess_plus['OffsetGroup'] = '超過リターン'
    df_excess_minus = df[['資産クラス', '資産超過リターン']].copy()
    df_excess_minus['リターン'] = df_excess_minus['資産超過リターン'].apply(lambda x: x if x < 0 else np.nan)
    df_excess_minus['凡例'] = '超過リターン(-)'
    df_excess_minus['OffsetGroup'] = '超過リターン'
    
    df_combined = pd.concat([df_p, df_b, df_excess_plus[['資産クラス', 'リターン', '凡例', 'OffsetGroup']], df_excess_minus[['資産クラス', 'リターン', '凡例', 'OffsetGroup']]])
    df_combined = df_combined.dropna(subset=['リターン'])
    
    offset_order = ['ポートフォリオ', 'ベンチマーク', '超過リターン']
    legend_order_combined = ['ポートフォリオ', 'ベンチマーク', '超過リターン(+)', '超過リターン(-)']
    color_scale = alt.Scale(domain=legend_order_combined, range=[COLOR_PORTFOLIO, COLOR_BENCHMARK, COLOR_POSITIVE, COLOR_NEGATIVE])

    base = alt.Chart(df_combined).encode(
        x=alt.X('資産クラス:N', sort=sort_order, axis=alt.Axis(title='資産クラス', labelAngle=-90, labelLimit=0)),
        y=alt.Y('リターン:Q', axis=alt.Axis(title=f"リターン（{return_label}）", format=".1%")),
        xOffset=alt.XOffset('OffsetGroup:N', sort=offset_order),
    )
    
    bars = base.mark_bar().encode(
        color=alt.Color('凡例:N', scale=color_scale, legend=alt.Legend(orient='top', title="凡例")),
        tooltip=[alt.Tooltip('資産クラス:N'), alt.Tooltip('OffsetGroup:N', title='種別'), alt.Tooltip('リターン:Q', format=".2%")]
    )

    # プラス方向のラベル（棒の上）
    text_plus = base.mark_text(align='center', baseline='bottom', dy=-5, fontSize=10).encode(
        text=alt.Text('リターン:Q', format='.1%')
    ).transform_filter(alt.datum.リターン >= 0)
    
    # マイナス方向のラベル（棒の下）
    text_minus = base.mark_text(align='center', baseline='top', dy=5, fontSize=10).encode(
        text=alt.Text('リターン:Q', format='.1%')
    ).transform_filter(alt.datum.リターン < 0)

    final_chart = alt.layer(bars, text_plus, text_minus).properties(title="資産クラス別 リターン比較", height=550).interactive()
    
    return final_chart

def plot_detailed_brinson_waterfall(summary, df_brinson):
    if df_brinson.empty or summary is None: return alt.Chart(pd.DataFrame()).mark_text(text="データなし").properties(height=600)

    df_brinson = df_brinson.dropna(subset=['資産クラス', '資産配分効果', '銘柄選択効果']).copy()
    df_brinson['資産クラス'] = df_brinson['資産クラス'].astype(str).str.strip()
    df_brinson = df_brinson[df_brinson['資産クラス'] != ""]
    if df_brinson.empty: return alt.Chart(pd.DataFrame()).mark_text(text="有効な要因データなし").properties(height=600)

    data = []
    data.append({"Category": "ベンチマーク", "Value": summary['Rb_fy'], "Type": "Benchmark"})
    
    df_brinson['資産配分効果_abs'] = df_brinson['資産配分効果'].abs()
    df_brinson['銘柄選択効果_abs'] = df_brinson['銘柄選択効果'].abs()

    df_brinson_sorted_aa = df_brinson.sort_values(['資産配分効果_abs', '資産クラス'], ascending=[False, True])
    for index, row in df_brinson_sorted_aa.iterrows():
        if abs(row['資産配分効果']) < 1e-9: continue
        category_name = f"{row['資産クラス']}（配分）"
        data.append({"Category": category_name, "Value": row['資産配分効果'], "Type": "AA"})
    
    df_brinson_sorted_ss = df_brinson.sort_values(['銘柄選択効果_abs', '資産クラス'], ascending=[False, True])
    for index, row in df_brinson_sorted_ss.iterrows():
        if abs(row['銘柄選択効果']) < 1e-9: continue
        category_name = f"{row['資産クラス']}（銘柄）"
        data.append({"Category": category_name, "Value": row['銘柄選択効果'], "Type": "SS"})

    factor_indices = [i for i, item in enumerate(data) if item['Type'] in ['AA', 'SS']]
    if abs(summary['Error']) > 1e-9:
        if factor_indices:
            last_factor_index = factor_indices[-1]
            data[last_factor_index]['Value'] += summary['Error']
        else:
            data.append({"Category": "調整項（誤差）", "Value": summary['Error'], "Type": "Other"})

    data.append({"Category": "ポートフォリオ", "Value": summary['Rp_fy'], "Type": "Portfolio"})
    
    df = pd.DataFrame(data)
    df['End'] = 0.0
    df['Start'] = 0.0
    
    current_level = 0.0
    for i in range(len(df)):
        value = df.loc[i, 'Value']
        if np.isnan(value): value = 0.0
        if df.loc[i, 'Type'] in ['Benchmark', 'Portfolio']:
            df.loc[i, 'Start'] = 0.0
            df.loc[i, 'End'] = value
            if df.loc[i, 'Type'] == 'Benchmark': current_level = value
        else:
            df.loc[i, 'Start'] = current_level
            current_level += value
            df.loc[i, 'End'] = current_level

    def get_color(row):
        if row['Type'] == "Benchmark": return COLOR_BENCHMARK
        if row['Type'] == "Portfolio": return COLOR_PORTFOLIO
        if row['Type'] == "AA": return COLOR_AA_POSITIVE if row['Value'] >= 0 else COLOR_AA_NEGATIVE
        return COLOR_POSITIVE if row['Value'] >= 0 else COLOR_NEGATIVE
        
    df['Color'] = df.apply(get_color, axis=1)

    y_min = df[['Start', 'End']].min().min()
    y_max = df[['Start', 'End']].max().max()
    y_range = y_max - y_min
    if y_range == 0 or np.isnan(y_range): y_range = 0.1

    threshold = y_range * 0.05
    label_offset = y_range * 0.03

    def calculate_label_props_detailed(row):
        is_small = abs(row['Value']) < threshold
        is_positive = row['Value'] >= 0
        if row['Type'] in ['Benchmark', 'Portfolio']:
            y_pos = row['End'] + (label_offset if row['End'] >= 0 else -label_offset)
            color = 'black'
        elif is_small:
            y_pos = row['End'] + (label_offset if is_positive else -label_offset)
            color = 'black'
        else:
            y_pos = (row['Start'] + row['End']) / 2
            color = 'white'
        return pd.Series([y_pos, color])

    df[['LabelYPosition', 'LabelColor']] = df.apply(calculate_label_props_detailed, axis=1)

    def format_value(row):
        if row['Type'] in ['Benchmark', 'Portfolio']: return f"{row['Value']:.2%}"
        else: return f"{row['Value']:+.2%}"
    df['ValueFormatted'] = df.apply(format_value, axis=1)

    df['NextCategory'] = df['Category'].shift(-1)
    df_connectors = df.dropna(subset=['NextCategory']).copy()
    
    sort_order = df['Category'].tolist()
    bar_size = max(15, 60 - len(df) * 2)

    chart = alt.Chart(df).mark_bar(size=bar_size).encode(
        x=alt.X('Category:N', sort=sort_order, axis=alt.Axis(title="要因", labelAngle=-90, labelLimit=0)),
        y=alt.Y('Start:Q', axis=alt.Axis(title='リターン/寄与度', format=".1%")),
        y2='End:Q', color=alt.Color('Color:N', scale=None),
        tooltip=[alt.Tooltip('Category:N', title='要因'), alt.Tooltip('Type:N', title='種別'), alt.Tooltip('Value:Q', format="+.4%", title='寄与度')]
    )
    
    text = alt.Chart(df).mark_text(align='center', baseline='middle', fontWeight='bold', fontSize=10).encode(
        x=alt.X('Category:N', sort=sort_order), y=alt.Y('LabelYPosition:Q'),
        color=alt.Color('LabelColor:N', scale=None), text=alt.Text('ValueFormatted:N')
    )

    rules = alt.Chart(df_connectors).mark_rule(strokeDash=[5, 5], strokeWidth=1, color='gray').encode(
        y='End:Q', x=alt.X('Category:N', sort=sort_order), x2=alt.X2('NextCategory:N')
    )

    final_chart = alt.layer(chart, text, rules).properties(title="パフォーマンス要因詳細分解（ブリンソン分解）", height=600).resolve_scale(y='shared').interactive()
    
    return final_chart

def plot_risk_return(summary, df_asset_performance, is_all_period):
    if summary is None: return alt.Chart(pd.DataFrame()).mark_text(text="データなし").properties(height=500)
    
    df_assets = df_asset_performance.copy()
    df_assets['凡例'] = '資産クラス'
    df_assets = df_assets.rename(columns={'Rp_fy': 'リターン', 'Risk_P_fy': 'リスク'})
    
    data_points = []
    if summary and not np.isnan(summary.get('Risk_P_fy')):
        data_points.append({'資産クラス': 'ポートフォリオ', 'リターン': summary['Rp_fy'], 'リスク': summary['Risk_P_fy'], '凡例': 'ポートフォリオ'})
    if summary and not np.isnan(summary.get('Risk_B_fy')):
        data_points.append({'資産クラス': '複合ベンチマーク', 'リターン': summary['Rb_fy'], 'リスク': summary['Risk_B_fy'], '凡例': '複合ベンチマーク'})
    df_points = pd.DataFrame(data_points)
    
    df_plot = pd.concat([df_assets[['資産クラス', 'リターン', 'リスク', '凡例']], df_points])
    df_plot = df_plot.dropna(subset=['リターン', 'リスク'])

    if df_plot.empty: return alt.Chart(pd.DataFrame()).mark_text(text="データなし").properties(height=500)

    if is_all_period:
        chart_title = "リスク・リターン散布図（年率換算）"
        y_axis_title = "リターン（年率）"
    else:
        chart_title = "リスク・リターン散布図（リスクは年率、リターンは期間累計）"
        y_axis_title = "リターン（期間累計）"

    chart = alt.Chart(df_plot).mark_point(size=100, filled=True, opacity=0.8).encode(
        x=alt.X('リスク:Q', axis=alt.Axis(title='リスク（標準偏差、年率）', format=".1%")),
        y=alt.Y('リターン:Q', axis=alt.Axis(title=y_axis_title, format=".1%")),
        color=alt.Color('凡例:N', scale=alt.Scale(domain=['資産クラス', 'ポートフォリオ', '複合ベンチマーク'], range=['#B0C4DE', COLOR_PORTFOLIO, COLOR_BENCHMARK])),
        shape=alt.Shape('凡例:N', scale=alt.Scale(domain=['資産クラス', 'ポートフォリオ', '複合ベンチマーク'], range=['circle', 'square', 'triangle'])),
        tooltip=[alt.Tooltip('資産クラス:N'), alt.Tooltip('リターン:Q', format=".2%"), alt.Tooltip('リスク:Q', format=".2%")]
    )
    
    text = chart.mark_text(align='left', baseline='middle', dx=10).encode(text='資産クラス:N')
    
    final_chart = (chart + text).properties(title=chart_title, height=500, padding={"right": 100}).interactive()
    
    return final_chart

def create_allocation_strategy_charts(df_brinson):
    if df_brinson.empty:
        empty_chart = alt.Chart(pd.DataFrame()).mark_text(text="データなし").properties(height=300)
        return empty_chart, empty_chart, empty_chart

    df = df_brinson.copy()
    COL_ACTION = 'AA_行動（Wp-Wb）'
    COL_ENVIRONMENT = 'AA_環境（Rb-R_total）'
    COL_EFFECT = '資産配分効果'

    df['Action_abs'] = df[COL_ACTION].abs()
    sort_order = df.sort_values('Action_abs', ascending=False)['資産クラス'].tolist()
    chart_height = max(300, len(df) * 45)

    Y_axis = alt.Y('資産クラス:N', sort=sort_order, axis=alt.Axis(title='資産クラス', labelLimit=0))
    Y_axis_hidden = alt.Y('資産クラス:N', sort=sort_order, axis=None)

    chart1_action = alt.Chart(df).mark_bar().encode(
        y=Y_axis,
        x=alt.X(f'{COL_ACTION}:Q', axis=alt.Axis(title='実際配分 - 基準配分 (Wp-Wb)', format=".1%")),
        color=alt.condition(alt.datum[COL_ACTION] > 0, alt.value(COLOR_STRATEGY_OVER), alt.value(COLOR_STRATEGY_UNDER)),
        tooltip=[alt.Tooltip('資産クラス:N'), alt.Tooltip('Wp_avg:Q', format=".1%", title='実際配分(Wp)'), alt.Tooltip('Wb_avg:Q', format=".1%", title='基準配分(Wb)'), alt.Tooltip(f'{COL_ACTION}:Q', format="+.1%", title='差異(Wp-Wb)')]
    ).properties(title="【行動】基準配分からの調整", height=chart_height)

    chart2_environment = alt.Chart(df).mark_bar().encode(
        y=Y_axis_hidden,
        x=alt.X(f'{COL_ENVIRONMENT}:Q', axis=alt.Axis(title='BMリターン - 全体平均 (Rb-R_total)', format=".1%")),
        color=alt.condition(alt.datum[COL_ENVIRONMENT] > 0, alt.value(COLOR_POSITIVE), alt.value(COLOR_NEGATIVE)),
        tooltip=[alt.Tooltip('資産クラス:N'), alt.Tooltip('Rb_fy:Q', format=".2%", title='BMリターン(Rb)'), alt.Tooltip(f'{COL_ENVIRONMENT}:Q', format="+.2%", title='全体平均との差')]
    ).properties(title="【環境】市場の有利/不利（平均比）", height=chart_height)

    chart3_evaluation = alt.Chart(df).mark_bar().encode(
        y=Y_axis_hidden,
        x=alt.X(f'{COL_EFFECT}:Q', axis=alt.Axis(title='資産配分効果（リターン寄与）', format=".1%")),
        color=alt.condition(alt.datum[COL_EFFECT] > 0, alt.value(COLOR_POSITIVE), alt.value(COLOR_NEGATIVE)),
        tooltip=[alt.Tooltip('資産クラス:N'), alt.Tooltip(f'{COL_EFFECT}:Q', format="+.2%")]
    ).properties(title="【評価】資産配分効果", height=chart_height)
    
    return chart1_action, chart2_environment, chart3_evaluation

def create_selection_strategy_charts(df_brinson):
    if df_brinson.empty:
        empty_chart = alt.Chart(pd.DataFrame()).mark_text(text="データなし").properties(height=300)
        return empty_chart, empty_chart, empty_chart

    df = df_brinson.copy()
    COL_ACTION = 'SS_行動（Rp-Rb）'
    COL_ENVIRONMENT = 'Wp_avg' 
    COL_EFFECT = '銘柄選択効果'
    
    df['Action_abs'] = df[COL_ACTION].abs()
    sort_order = df.sort_values('Action_abs', ascending=False)['資産クラス'].tolist()
    chart_height = max(300, len(df) * 45)

    Y_axis = alt.Y('資産クラス:N', sort=sort_order, axis=alt.Axis(title='資産クラス', labelLimit=0))
    Y_axis_hidden = alt.Y('資産クラス:N', sort=sort_order, axis=None)

    chart1_action = alt.Chart(df).mark_bar().encode(
        y=Y_axis,
        x=alt.X(f'{COL_ACTION}:Q', axis=alt.Axis(title='ポートフォリオ - BM (Rp-Rb)', format=".1%")),
        color=alt.condition(alt.datum[COL_ACTION] > 0, alt.value(COLOR_POSITIVE), alt.value(COLOR_NEGATIVE)),
        tooltip=[alt.Tooltip('資産クラス:N'), alt.Tooltip('Rp_fy:Q', format=".2%", title='ポートフォリオ(Rp)'), alt.Tooltip('Rb_fy:Q', format=".2%", title='ベンチマーク(Rb)'), alt.Tooltip(f'{COL_ACTION}:Q', format="+.2%", title='差異(Rp-Rb)')]
    ).properties(title="【行動】超過リターンの獲得", height=chart_height)

    chart2_environment = alt.Chart(df).mark_bar().encode(
        y=Y_axis_hidden,
        x=alt.X(f'{COL_ENVIRONMENT}:Q', axis=alt.Axis(title='実際配分比率 (Wp)', format=".1%")),
        color=alt.Color(f'{COL_ENVIRONMENT}:Q', scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=[alt.Tooltip('資産クラス:N'), alt.Tooltip(f'{COL_ENVIRONMENT}:Q', format=".1%", title='実際配分(Wp)')]
    ).properties(title="【環境】実際配分比率（重要度）", height=chart_height)

    chart3_evaluation = alt.Chart(df).mark_bar().encode(
        y=Y_axis_hidden,
        x=alt.X(f'{COL_EFFECT}:Q', axis=alt.Axis(title='銘柄選択効果（リターン寄与）', format=".1%")),
        color=alt.condition(alt.datum[COL_EFFECT] > 0, alt.value(COLOR_POSITIVE), alt.value(COLOR_NEGATIVE)),
        tooltip=[alt.Tooltip('資産クラス:N'), alt.Tooltip(f'{COL_EFFECT}:Q', format="+.2%")]
    ).properties(title="【評価】銘柄選択効果", height=chart_height)
    
    return chart1_action, chart2_environment, chart3_evaluation


# --- 6. Streamlit UI 構築 ---

def main():
    st.sidebar.title("設定パネル")

    st.sidebar.header("1. データ準備")
    
    @st.cache_data
    def get_sample_data(): return generate_sample_data()

    sample_data = get_sample_data()

    st.sidebar.download_button(
        label="サンプルデータ (Excel) をダウンロード",
        data=sample_data, file_name="sample_portfolio_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    uploaded_file = st.sidebar.file_uploader("分析データ（Excel）をアップロード", type="xlsx", key="file_uploader")

    if "analysis_results" not in st.session_state: st.session_state.analysis_results = None
    if "last_file_name" not in st.session_state: st.session_state.last_file_name = None

    st.title("有価証券パフォーマンス分析ダッシュボード")

    data_source_used = None
    if uploaded_file:
        data_source_used = uploaded_file
        data_source_name = "アップロードデータ"
        if uploaded_file.name != st.session_state.last_file_name:
            st.session_state.analysis_results = None
            st.session_state.last_file_name = uploaded_file.name
    else:
        data_source_used = io.BytesIO(sample_data)
        data_source_name = "サンプルデータ"
        if st.session_state.last_file_name is not None:
             st.session_state.analysis_results = None
             st.session_state.last_file_name = None

    if data_source_used:
        df_portfolio, df_benchmark, df_policy_mix, asset_order, error = load_data(data_source_used)
    else:
        error = "データを準備してください。"
        df_portfolio, df_benchmark, df_policy_mix, asset_order = None, None, None, []
    
    if error:
        st.error(error)
    elif df_portfolio is not None:
        df_monthly = calculate_monthly_returns(df_portfolio, FISCAL_START_MONTH)
        
        available_fys_num = sorted(df_monthly.dropna(subset=['月次リターン（Rp）'])['年度'].unique())
        available_fys_str = [str(int(fy)) for fy in available_fys_num if not np.isnan(fy)]
        
        if not available_fys_str:
            st.warning("分析可能な年度データがありません。")
        else:
            analysis_options = ["全期間"] + sorted(available_fys_str, reverse=True)
            
            all_assets_original = asset_order.copy()
            for a in df_monthly['資産クラス'].unique():
                if a not in all_assets_original and not pd.isna(a):
                    all_assets_original.append(a)
            default_exclude = [a for a in all_assets_original if any(p in a for p in DEFAULT_EXCLUDE_PATTERN)]

            with st.sidebar.form(key='analysis_form'):
                st.header("2. 分析設定")
                selected_option = st.selectbox("分析対象期間を選択", analysis_options)
                
                rebalance_option = st.selectbox(
                    "ベンチマークのリバランス方針", 
                    ["月次リバランス", "リバランスなし（Buy & Hold）", "カスタム（年次リバランス）"]
                )
                
                excluded_assets = st.multiselect("評価から除外する資産クラス", all_assets_original, default=default_exclude)
                submit_button = st.form_submit_button(label='分析実行')

            if submit_button:
                is_all_period = (selected_option == "全期間")
                selected_fy = None if is_all_period else int(selected_option)
                analysis_period_label = "全期間" if is_all_period else f"{selected_fy}年度"
                
                results, df_analysis_monthly, error_msg = run_analysis(df_monthly, df_benchmark, df_policy_mix, selected_fy, excluded_assets, is_all_period, rebalance_option)
                
                st.session_state.analysis_results = {
                    "results": results, "label": analysis_period_label, "is_all_period": is_all_period,
                    "error_msg": error_msg, "rebalance_option": rebalance_option,
                    "asset_order": asset_order, 
                    "intermediate": {"df_monthly": df_monthly, "df_analysis_monthly": df_analysis_monthly},
                    "raw_input": {"df_portfolio": df_portfolio, "df_benchmark": df_benchmark, "df_policy_mix": df_policy_mix}
                }

            if st.session_state.analysis_results:
                results_data = st.session_state.analysis_results
                
                if results_data["results"]:
                     summary, df_monthly_ts, df_brinson, df_asset_performance = results_data["results"]
                else:
                     summary, df_monthly_ts, df_brinson, df_asset_performance = None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

                analysis_period_label = results_data["label"]
                is_all_period = results_data["is_all_period"]
                rebalance_option_used = results_data.get("rebalance_option", "不明")
                saved_asset_order = results_data.get("asset_order", [])
                error_msg = results_data.get("error_msg")
                df_analysis_monthly = results_data["intermediate"]["df_analysis_monthly"]

                st.header(f"{analysis_period_label} パフォーマンス分析結果")
                st.markdown(f"データソース: {data_source_name} | **BM方針: {rebalance_option_used}**")
                
                if error_msg: st.warning(error_msg)

                return_label = "年率" if is_all_period else "期間"

                if summary: insights = generate_insights(summary, df_brinson, return_label, analysis_period_label)
                else: insights = []

                tab1, tab2, tab3, tab4, tab5 = st.tabs(["エグゼクティブサマリー", "資産クラス分析", "要因分解・戦略評価", "リスク分析", "データ詳細"])
                
                with tab1:
                    if summary:
                        st.subheader("エグゼクティブサマリー")
                        st.markdown("#### 全体パフォーマンス概要")
                        col1, col2, col3 = st.columns(3)
                        col1.metric(f"ポートフォリオ リターン（{return_label}）", f"{summary['Rp_fy']:.2%}")
                        col2.metric(f"複合ベンチマーク リターン（{return_label}）", f"{summary['Rb_fy']:.2%}")
                        
                        delta_color = "normal" if summary['Excess_Return_fy'] >= 0 else "inverse"
                        col3.metric(f"超過リターン（{return_label}）", f"{summary['Excess_Return_fy']:.2%}", delta=f"{summary['Excess_Return_fy']:+.2%}", delta_color=delta_color)

                        st.altair_chart(plot_cumulative_returns_with_excess(df_monthly_ts), use_container_width=True)
                        st.caption("※累積リターン（折れ線、左軸）と、月次の超過リターン（棒、右軸）を表示しています。両軸の0%の位置は揃えています。")
                        
                        st.altair_chart(plot_brinson_waterfall(summary), use_container_width=True)
                        st.caption("※資産配分効果（緑/オレンジ）、銘柄選択効果（青/赤）で色分けしています。")
                        
                        st.markdown("---")
                        st.markdown("#### 分析からの示唆・キーメッセージ")
                        if insights:
                            for insight in insights: st.markdown(f"- {insight}")
                        else:
                            st.info("表示する示唆はありません。")
                        
                        st.markdown("---")
                        with st.expander("リターン計算方法と要因分析の詳細について"):
                            st.info(f"""
                            **リターン計算方法について:**
                            ポートフォリオの実績リターンは、キャッシュフロー（元本の増減）の影響を除外した**時間加重収益率（TWRR）**の近似値を用いています。具体的には、「会計ベースの収益額」と「修正ディーツ法」を組み合わせて計算しています。

                            **要因分析サマリー (Brinson-Fachlerモデル近似):**
                            * 資産配分効果（計算値）: {summary['AA_total']:.4%}
                            * 銘柄選択効果（計算値、相互作用効果含む）: {summary['SS_total']:.4%}
                            * 計算誤差（※）: {summary['Error']:.4%}
                            """)

                with tab2:
                    if summary:
                        st.subheader("資産クラス別 パフォーマンス分析")
                        
                        st.markdown("#### 実績とベンチマークの構成比推移比較")
                        
                        st.altair_chart(plot_latest_allocation_pie_charts(df_analysis_monthly, saved_asset_order), use_container_width=True)
                        st.altair_chart(plot_allocation_bar_chart(df_analysis_monthly, saved_asset_order), use_container_width=True)
                        
                        st.markdown("##### 構成比推移のデータテーブル")
                        col_t1, col_t2 = st.columns(2)
                        
                        df_pivot_wp = df_analysis_monthly.pivot_table(index='年月', columns='資産クラス', values='実際配分比率（Wp）', fill_value=0)
                        df_pivot_wb = df_analysis_monthly.pivot_table(index='年月', columns='資産クラス', values='基準配分比率（Wb）', fill_value=0)
                        
                        cols_wp = [c for c in saved_asset_order if c in df_pivot_wp.columns]
                        cols_wb = [c for c in saved_asset_order if c in df_pivot_wb.columns]
                        df_pivot_wp = df_pivot_wp[cols_wp]
                        df_pivot_wb = df_pivot_wb[cols_wb]
                        
                        df_pivot_wp.index = df_pivot_wp.index.strftime('%Y/%m')
                        df_pivot_wb.index = df_pivot_wb.index.strftime('%Y/%m')

                        with col_t1:
                            st.markdown("**実績構成比 (Wp)**")
                            st.dataframe(df_pivot_wp.style.format("{:.2%}"), use_container_width=True)
                        with col_t2:
                            st.markdown("**ベンチマーク構成比 (Wb)**")
                            st.dataframe(df_pivot_wb.style.format("{:.2%}"), use_container_width=True)

                        st.caption(f"※BM方針（{rebalance_option_used}）による構成比の変化を詳細に確認できます。表の右上からCSVのダウンロードが可能です。")
                        
                        st.markdown("---")

                        st.markdown("#### 資産クラス別 リターン比較")
                        st.altair_chart(plot_asset_returns_comparison(df_asset_performance, return_label, saved_asset_order), use_container_width=True)
                        st.caption(f"※各資産クラスの「ポートフォリオ（濃紺）」「ベンチマーク（灰）」「超過リターン（青/赤）」（いずれも{return_label}ベース）を比較しています。超過リターンが大きい順に並べています。")

                with tab3:
                    if summary:
                        st.subheader("パフォーマンス要因分解と運用戦略の評価")
                        st.markdown("##### 全体要因分析サマリー")
                        col1, col2, col3 = st.columns(3)
                        
                        col1.metric(f"超過リターン（{return_label}）", f"{summary['Excess_Return_fy']:.2%}")
                        col2.metric("資産配分効果（全体）", f"{summary['AA_total']:.2%}")
                        ss_with_error = summary['SS_total'] + summary['Error']
                        col3.metric("銘柄選択効果（全体, 誤差含）", f"{ss_with_error:.2%}")
                        st.markdown("---")

                        st.markdown("#### 1. 資産配分戦略の評価： 行動 × 環境 = 評価")
                        st.markdown("計算構造: `(実際配分Wp - 基準配分Wb) × (資産クラス別BMリターンRb - 全体BMリターンR_total)`")
                        
                        chart_aa1, chart_aa2, chart_aa3 = create_allocation_strategy_charts(df_brinson)
                        col_aa1, col_aa2, col_aa3 = st.columns([1.2, 1, 1])
                        with col_aa1: st.altair_chart(chart_aa1, use_container_width=True)
                        with col_aa2: st.altair_chart(chart_aa2, use_container_width=True)
                        with col_aa3: st.altair_chart(chart_aa3, use_container_width=True)

                        st.caption("""
                        **見方:** 例えば、「行動」でオーバーウェイト（緑）した資産が、「環境」で市場平均より良好（青）だった場合、「評価」はプラス（青）になります（戦略成功）。
                        逆に、「行動」でアンダーウェイト（黄）した資産が、「環境」で市場平均より不良（赤）だった場合も、「評価」はプラス（青）になります（リスク回避成功）。
                        """)
                        st.markdown("---")

                        st.markdown("#### 2. 銘柄選択戦略の評価： 行動 × 環境 = 評価")
                        st.markdown("計算構造: `(ポートフォリオリターンRp - BMリターンRb) × 実際配分Wp`")

                        chart_ss1, chart_ss2, chart_ss3 = create_selection_strategy_charts(df_brinson)
                        col_ss1, col_ss2, col_ss3 = st.columns([1.2, 1, 1])
                        with col_ss1: st.altair_chart(chart_ss1, use_container_width=True)
                        with col_ss2: st.altair_chart(chart_ss2, use_container_width=True)
                        with col_ss3: st.altair_chart(chart_ss3, use_container_width=True)
                        
                        st.caption("""
                        **見方:** 「行動」でベンチマークを上回った（青）資産について、「環境（実際配分）」のウェイトが大きい（色が濃い）ほど、「評価」のプラス（青）が大きくなります。
                        """)
                        st.markdown("---")

                        st.markdown("#### 3. パフォーマンス要因詳細分解（ブリンソン分解）")
                        st.altair_chart(plot_detailed_brinson_waterfall(summary, df_brinson), use_container_width=True)
                        st.caption(f"※ベンチマークリターン（灰）から開始し、資産配分効果（緑：プラス、オレンジ：マイナス）、銘柄選択効果（青：プラス、赤：マイナス）を積み上げ、ポートフォリオのリターン（濃紺、{return_label}ベース）に至るまでを分解しています。")

                with tab4:
                    if summary:
                        st.subheader("リスク分析")
                        
                        has_risk_p = not np.isnan(summary['Risk_P_fy'])
                        has_risk_b = not np.isnan(summary['Risk_B_fy'])

                        if not has_risk_p or not has_risk_b:
                                st.warning("リスク計算（標準偏差）には2ヶ月以上の有効なデータが必要です。")
                        
                        col1, col2, col3 = st.columns(3)
                        risk_p_display = f"{summary['Risk_P_fy']:.2%}" if has_risk_p else "N/A"
                        risk_b_display = f"{summary['Risk_B_fy']:.2%}" if has_risk_b else "N/A"
                        col1.metric("ポートフォリオ リスク（年率）", risk_p_display)
                        col2.metric("複合ベンチマーク リスク（年率）", risk_b_display)
                        
                        sharpe_ratio_display = "N/A"
                        if has_risk_p and summary['Risk_P_fy'] > 0:
                            sharpe_ratio = summary['Rp_fy'] / summary['Risk_P_fy']
                            sharpe_ratio_display = f"{sharpe_ratio:.2f}"
                        col3.metric(f"シャープレシオ（{return_label}ベース, Rf=0）", sharpe_ratio_display)

                        st.altair_chart(plot_risk_return(summary, df_asset_performance, is_all_period), use_container_width=True)

                with tab5:
                    st.subheader("データ詳細・ダウンロードセンター")
                    st.markdown("#### 分析データ一括ダウンロード")
                    
                    intermediate_data = results_data.get("intermediate", {})
                    raw_input_data = results_data.get("raw_input", {})

                    if not df_brinson.empty:
                        df_brinson_dl = df_brinson.drop(columns=['Action_abs', '資産配分効果_abs', '銘柄選択効果_abs'], errors='ignore')
                    else:
                        df_brinson_dl = pd.DataFrame()

                    download_data = {
                        "1_入力_ポートフォリオ実績": raw_input_data.get("df_portfolio"),
                        "1_入力_BMリターン": raw_input_data.get("df_benchmark"),
                        "1_入力_基準配分": raw_input_data.get("df_policy_mix"),
                        "2_加工_月次リターン計算後(全期間)": intermediate_data.get("df_monthly"),
                        "3_加工_分析対象データ(対象期間)": intermediate_data.get("df_analysis_monthly"),
                        "4_結果_ブリンソン分解(対象期間)": df_brinson_dl,
                        "4_結果_全体時系列(対象期間)": df_monthly_ts,
                    }

                    excel_file = convert_dfs_to_excel(download_data)
                    if excel_file:
                            st.download_button(
                            label="全データ（入力・加工・結果）をExcelでダウンロード",
                            data=excel_file,
                            file_name=f"PortfolioAnalysis_Data_{analysis_period_label.replace('/', '-')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    st.markdown("---")
                    st.markdown("#### 分析結果サマリー（ブリンソン分解）プレビュー")

                    if not df_brinson_dl.empty:
                        columns_to_drop_display = ['AA_行動（Wp-Wb）', 'AA_環境（Rb-R_total）', 'SS_行動（Rp-Rb）']
                        df_brinson_display = df_brinson_dl.drop(columns=columns_to_drop_display, errors='ignore')

                        df_brinson_display['資産クラス'] = pd.Categorical(df_brinson_display['資産クラス'], categories=saved_asset_order, ordered=True)
                        df_brinson_display = df_brinson_display.sort_values('資産クラス')

                        st.dataframe(df_brinson_display.style.format({
                            'Wp_avg': '{:.2%}', 'Wb_avg': '{:.2%}',
                            'Rp_fy': '{:.2%}', 'Rb_fy': '{:.2%}', 'Risk_P_fy': '{:.2%}',
                            '資産配分効果': '{:.4%}', '銘柄選択効果': '{:.4%}', 'トータル効果': '{:.4%}'
                        }, na_rep="N/A"), use_container_width=True)
                        st.caption(f"※表示されているリターン（Rp_fy, Rb_fy）および各効果は「{return_label}」ベースです。リスク（Risk_P_fy）は「年率」ベースです。")
                    else:
                            st.info("データがありません。")
            
            elif not submit_button:
                 st.info("サイドバーで分析対象期間を選択し、「分析実行」ボタンを押して分析を開始してください。（設定変更後もボタン押下が必要です）")

    else:
        st.info("サイドバーから分析データ（Excel）をアップロードするか、サンプルデータをご利用ください。")


if __name__ == "__main__":
    main()