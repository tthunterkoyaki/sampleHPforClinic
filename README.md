import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.ticker as mticker

# --------------------------------------------------------------------------------
# 設定・初期化
# --------------------------------------------------------------------------------

# 日本語フォント設定（環境に合わせて調整）
# plt.rcParams['font.family'] = 'MS Gothic' 

# 乱数シード固定
np.random.seed(42)

# --------------------------------------------------------------------------------
# 1. データ生成関数
# --------------------------------------------------------------------------------

def generate_dummy_data(n_contracts=100):
    """ダミーの契約データと為替レート実績データを生成"""
    
    # --- 為替レート実績データ ---
    start_date = datetime(2023, 1, 1)
    today = datetime(2025, 12, 3)
    dates = pd.date_range(start=start_date, end=today, freq='B')
    n_days = len(dates)

    def generate_fx_series(start_price, volatility, n_days, seed):
        rng = np.random.default_rng(seed=seed)
        returns = rng.normal(0, volatility / np.sqrt(252), n_days)
        return start_price * np.exp(np.cumsum(returns))

    historical_fx_df = pd.DataFrame({
        'Date': dates,
        'USDJPY': generate_fx_series(150, 0.10, n_days, seed=42),
        'EURJPY': generate_fx_series(160, 0.08, n_days, seed=43),
        'CNHJPY': generate_fx_series(20, 0.05, n_days, seed=44)
    }).set_index('Date')

    # --- 契約データ ---
    contracts = []
    rng_c = np.random.default_rng(seed=42)

    for i in range(n_contracts):
        contract_date = start_date + timedelta(days=int(rng_c.integers(0, (today - start_date).days)))
        duration_years = rng_c.integers(1, 6)
        end_date_c = contract_date + relativedelta(years=duration_years)
        first_settlement_date = contract_date + relativedelta(months=1)

        # 輸入/輸出決定
        if rng_c.random() > 0.5:
            receive, pay, direction = rng_c.choice(['USD', 'EUR', 'CNH']), 'JPY', 'Import'
        else:
            receive, pay, direction = 'JPY', rng_c.choice(['USD', 'EUR', 'CNH']), 'Export'

        # タイプ決定
        d_type = rng_c.choice(['NonKO', 'NormalKO', 'TargetKO_Amount', 'TargetKO_Count'], p=[0.4, 0.3, 0.2, 0.1])

        contract = {
            '契約ID': f'C{i+1:04d}',
            '契約日': contract_date,
            '終了日': end_date_c,
            '初回決済日': first_settlement_date,
            '消滅日': np.nan,
            '収益': rng_c.integers(100, 1000) * 10000,
            '受取通貨': receive, '支払通貨': pay,
            '通常KO型': 1 if d_type == 'NormalKO' else 0,
            'ターゲットKO型': 1 if 'TargetKO' in d_type else 0,
            'KO・KI相場(1行目)': np.nan, 'KO・KI判定(1行目)': np.nan,
            '年間交換回数': rng_c.choice([1, 2, 4, 12]),
            '交換日': rng_c.choice([15, 25, '末']),
            '消滅判定開始日（ウィンドウ）': np.nan,
            'ターゲット金額（円）': np.nan, 'ターゲット回数（件数）': np.nan,
            '決済相場1': np.nan, '決済相場2': np.nan,
            '受取': rng_c.integers(1, 100) * 10000,
            '支払件数(特殊形の抽出後)': np.nan,
            '_Type': d_type
        }

        # レート設定
        base_rate = 150 if 'USD' in [receive, pay] else (160 if 'EUR' in [receive, pay] else 20)

        if d_type == 'NormalKO':
            factor = (1 + rng_c.uniform(0.05, 0.20)) if direction == 'Import' else (1 - rng_c.uniform(0.05, 0.20))
            contract['KO・KI相場(1行目)'] = base_rate * factor
            contract['KO・KI判定(1行目)'] = rng_c.choice(['ｱﾒﾘｶﾝ', 'ﾖｰﾛﾋﾟｱﾝ'])

        elif 'TargetKO' in d_type:
            contract['決済相場1'] = base_rate * rng_c.uniform(0.90, 1.10)
            if rng_c.random() > 0.7:
                contract['決済相場2'] = base_rate * rng_c.uniform(0.90, 1.10)
                contract['支払件数(特殊形の抽出後)'] = rng_c.integers(3, 12)
            
            if d_type == 'TargetKO_Amount':
                contract['ターゲット金額（円）'] = contract['受取'] * rng_c.uniform(5, 25)
            else:
                contract['ターゲット回数（件数）'] = rng_c.integers(5, 20)

        # ウィンドウ設定
        if rng_c.random() < 0.1:
            w_start = first_settlement_date + relativedelta(months=int(rng_c.integers(1, 12)))
            if w_start < end_date_c: contract['消滅判定開始日（ウィンドウ）'] = w_start

        # 過去消滅設定
        if rng_c.random() < 0.1 and contract['契約日'] < (today - relativedelta(months=3)):
            if first_settlement_date < today:
                days = (min(end_date_c, today) - first_settlement_date).days
                if days > 0:
                    contract['消滅日'] = first_settlement_date + timedelta(days=int(rng_c.integers(0, days)))

        contracts.append(contract)

    df = pd.DataFrame(contracts)
    for col in ['契約日', '終了日', '初回決済日', '消滅日', '消滅判定開始日（ウィンドウ）']:
        df[col] = pd.to_datetime(df[col])

    return df, historical_fx_df

# --------------------------------------------------------------------------------
# 2. ユーティリティ関数
# --------------------------------------------------------------------------------

def get_exchange_dates(start_date, end_date, annual_freq, exchange_day_rule):
    """初回決済日から終了日までの交換日リストを生成"""
    if annual_freq <= 0 or pd.isna(annual_freq) or pd.isna(start_date) or pd.isna(end_date):
        return []

    interval = 12 // annual_freq
    curr = pd.to_datetime(start_date).replace(day=1)
    end = pd.to_datetime(end_date)
    dates = []

    while curr <= end:
        if exchange_day_rule == '末':
            tgt = curr + relativedelta(day=31)
        else:
            try:
                tgt = curr + relativedelta(day=int(exchange_day_rule))
            except:
                curr += relativedelta(months=interval)
                continue
        
        if start_date <= tgt <= end:
            dates.append(tgt)
        curr += relativedelta(months=interval)
    
    return sorted(list(set(dates)))

def get_rate_at(fx_df, date, pair):
    """指定日のレート取得（直近過去営業日参照）"""
    if pair not in fx_df.columns: return np.nan
    try:
        return fx_df[pair].asof(date)
    except:
        return np.nan

# --------------------------------------------------------------------------------
# 3. 為替シミュレーション関数
# --------------------------------------------------------------------------------

def run_fx_simulation(hist_df, sim_start, sim_end, method='GBM', target_rates=None, n_sims=100, seed=42):
    """過去データからパラメータ推定し、将来レートを生成して結合する"""
    sim_dates = pd.date_range(start=sim_start, end=sim_end, freq='B')
    n_days = len(sim_dates)
    currencies = [c for c in hist_df.columns if 'JPY' in c]
    
    # パラメータ推定 (GBM用)
    params = {}
    for curr in currencies:
        log_ret = np.log(hist_df[curr] / hist_df[curr].shift(1)).dropna()
        params[curr] = {
            'mu': log_ret.mean() * 252,
            'sigma': log_ret.std() * np.sqrt(252)
        }

    # シミュレーション実行
    rng = np.random.default_rng(seed=seed)
    last_hist_date = pd.to_datetime(sim_start) - timedelta(days=1)
    last_prices = {c: hist_df[c].asof(last_hist_date) for c in currencies}
    
    all_scenarios = []

    for _ in range(n_sims):
        sim_data = {}
        for curr in currencies:
            if curr not in last_prices or pd.isna(last_prices[curr]): continue
            
            p0 = last_prices[curr]
            
            if method == 'GBM':
                mu, sigma = params[curr]['mu'], params[curr]['sigma']
                dt = 1/252
                drift = (mu - 0.5 * sigma**2) * dt
                diffusion = sigma * np.sqrt(dt) * rng.normal(0, 1, n_days)
                prices = p0 * np.exp(np.cumsum(drift + diffusion))
                
            elif method == 'Linear':
                target = target_rates.get(curr, p0) if target_rates else p0
                prices = np.linspace(p0, target, n_days+1)[1:]
            
            sim_data[curr] = prices
            
        sim_df = pd.DataFrame(sim_data, index=sim_dates)
        combined = pd.concat([hist_df, sim_df]).sort_index().ffill()
        all_scenarios.append(combined)
        
    return all_scenarios

# --------------------------------------------------------------------------------
# 4. KO判定ロジック関数
# --------------------------------------------------------------------------------

def check_normal_ko(contract, fx_df, start_date, pair, is_import):
    """通常KO型の判定"""
    ko_rate = contract['KO・KI相場(1行目)']
    if pd.isna(ko_rate): return None
    
    end_date = contract['終了日']
    
    # 条件判定関数
    def is_out(r): return r >= ko_rate if is_import else r <= ko_rate

    if contract['KO・KI判定(1行目)'] == 'ｱﾒﾘｶﾝ':
        # 日次判定
        rates = fx_df[pair].loc[start_date:end_date]
        if rates.empty: return None
        mask = is_out(rates)
        if mask.any():
            return mask[mask].index[0]
            
    elif contract['KO・KI判定(1行目)'] == 'ﾖｰﾛﾋﾟｱﾝ':
        # 交換日のみ判定
        ex_dates = get_exchange_dates(contract['初回決済日'], end_date, contract['年間交換回数'], contract['交換日'])
        for d in [d for d in ex_dates if d >= start_date]:
            r = get_rate_at(fx_df, d, pair)
            if pd.notna(r) and is_out(r):
                return d
    return None

def check_target_ko(contract, fx_df, start_date, pair):
    """ターゲットKO型（金額・回数）の判定"""
    target_amt = contract['ターゲット金額（円）']
    target_cnt = contract['ターゲット回数（件数）']
    
    if pd.isna(target_amt) and pd.isna(target_cnt): return None
    
    ex_dates = get_exchange_dates(contract['初回決済日'], contract['終了日'], contract['年間交換回数'], contract['交換日'])
    
    acc_val = 0
    cnt = 0
    rate1 = contract['決済相場1']
    rate2 = contract['決済相場2']
    switch_cnt = contract['支払件数(特殊形の抽出後)']
    
    for d in ex_dates:
        cnt += 1
        if d < start_date: continue # 過去分はスキップ（カウントは進む）
        
        curr_target = rate2 if (pd.notna(rate2) and pd.notna(switch_cnt) and cnt > switch_cnt) else rate1
        if pd.isna(curr_target): continue
        
        r = get_rate_at(fx_df, d, pair)
        if pd.isna(r): continue
        
        # ターゲットレートを超えたら蓄積
        if r > curr_target:
            if pd.notna(target_amt):
                acc_val += (r - curr_target) * contract['受取']
            elif pd.notna(target_cnt):
                acc_val += 1
        
        threshold = target_amt if pd.notna(target_amt) else target_cnt
        if pd.notna(threshold) and acc_val >= threshold:
            return d
            
    return None

def evaluate_contract_ko(contract, fx_df):
    """契約のKO判定メイン関数"""
    if pd.notna(contract['消滅日']): return contract['消滅日']
    
    # 通貨ペア特定
    ccy = set([contract['受取通貨'], contract['支払通貨']])
    pair = 'USDJPY' if 'USD' in ccy else ('EURJPY' if 'EUR' in ccy else ('CNHJPY' if 'CNH' in ccy else None))
    if not pair: return None

    # 判定開始日
    judgement_start = max(contract['初回決済日'], contract.get('消滅判定開始日（ウィンドウ）', pd.NaT))
    if pd.isna(judgement_start): judgement_start = contract['初回決済日']
    
    if judgement_start > contract['終了日']: return None

    # タイプ別分岐
    ko_date = None
    if contract['通常KO型'] == 1:
        is_imp = contract['受取通貨'] != 'JPY'
        ko_date = check_normal_ko(contract, fx_df, judgement_start, pair, is_imp)
    elif contract['ターゲットKO型'] == 1:
        ko_date = check_target_ko(contract, fx_df, judgement_start, pair)
        
    return ko_date

# --------------------------------------------------------------------------------
# 5. 収益計算プロセス関数
# --------------------------------------------------------------------------------

def calculate_profits(contracts_df, fx_scenarios, config):
    """全シナリオ・全契約の収益計算"""
    ana_start = pd.to_datetime(config['analysis_start_date'])
    ana_end = pd.to_datetime(config['analysis_end_date'])
    rr_non_ko = config['roll_rate_non_ko']
    rr_ko = config['roll_rate_ko']
    
    results = [] # 各シナリオの合計収益
    all_details = [] # 詳細ログ
    
    print(f"Calculating profits for {len(fx_scenarios)} scenarios...")

    for i, fx_df in enumerate(fx_scenarios):
        if (i+1) % 50 == 0: print(f"Processing {i+1}/{len(fx_scenarios)}")
        
        scenario_total = 0
        scenario_details = []
        
        # DataFrameの行を辞書としてイテレート
        for _, contract in contracts_df.iterrows():
            cid = contract['契約ID']
            base_profit = contract['収益']
            if pd.isna(base_profit): continue
            
            # 1. 新規契約収益
            if ana_start <= contract['契約日'] <= ana_end:
                scenario_total += base_profit
                scenario_details.append({'Scenario': i+1, '契約ID': cid, 'Type': 'New', 'Profit': base_profit, 'Date': contract['契約日']})
            
            # 2. ロール収益判定
            
            # A. 過去に消滅済み
            if pd.notna(contract['消滅日']):
                if ana_start <= contract['消滅日'] <= ana_end:
                    p = base_profit * rr_ko
                    scenario_total += p
                    scenario_details.append({'Scenario': i+1, '契約ID': cid, 'Type': 'Roll(HistKO)', 'Profit': p, 'Date': contract['消滅日']})
                continue
                
            # B. KO条件なし -> 満期判定
            if contract['通常KO型'] == 0 and contract['ターゲットKO型'] == 0:
                if ana_start <= contract['終了日'] <= ana_end:
                    p = base_profit * rr_non_ko
                    scenario_total += p
                    scenario_details.append({'Scenario': i+1, '契約ID': cid, 'Type': 'Roll(Mat)', 'Profit': p, 'Date': contract['終了日']})
                continue
            
            # C. KO条件あり -> シミュレーション判定
            ko_date = evaluate_contract_ko(contract, fx_df)
            
            if ko_date:
                # KO発生
                if ana_start <= ko_date <= ana_end:
                    p = base_profit * rr_ko
                    scenario_total += p
                    scenario_details.append({'Scenario': i+1, '契約ID': cid, 'Type': 'Roll(SimKO)', 'Profit': p, 'Date': ko_date})
            else:
                # 満期到達
                if ana_start <= contract['終了日'] <= ana_end:
                    p = base_profit * rr_non_ko
                    scenario_total += p
                    scenario_details.append({'Scenario': i+1, '契約ID': cid, 'Type': 'Roll(SimMat)', 'Profit': p, 'Date': contract['終了日']})
        
        results.append(scenario_total)
        all_details.extend(scenario_details)
        
    return results, pd.DataFrame(all_details)

def calc_ko_probabilities(contracts_df, details_df, n_sims):
    """消滅確率の計算"""
    if details_df.empty: return pd.Series()
    
    # シミュレーションでKOしたレコードを抽出
    sim_kos = details_df[details_df['Type'] == 'Roll(SimKO)']
    counts = sim_kos['契約ID'].value_counts()
    
    # 対象契約（未消滅かつKO型）
    targets = contracts_df[
        contracts_df['消滅日'].isna() & 
        ((contracts_df['通常KO型']==1) | (contracts_df['ターゲットKO型']==1))
    ]['契約ID']
    
    probs = {}
    for cid in targets:
        probs[cid] = counts.get(cid, 0) / n_sims if n_sims > 0 else 0
        
    return pd.Series(probs).sort_values(ascending=False)

# --------------------------------------------------------------------------------
# 6. メイン実行ブロック
# --------------------------------------------------------------------------------

if __name__ == '__main__':
    # --- 1. データ準備 ---
    contracts_df, historical_fx_df = generate_dummy_data(n_contracts=500)
    
    # --- 2. 設定 ---
    config = {
        'analysis_start_date': '2025-04-01',
        'analysis_end_date': '2027-03-31',
        'roll_rate_non_ko': 0.6,
        'roll_rate_ko': 0.7,
        'simulation_method': 'GBM',
        'n_simulations': 100
    }
    
    # シミュレーション期間決定（実績データ翌日～分析終了日）
    sim_start = historical_fx_df.index.max() + timedelta(days=1)
    sim_end = pd.to_datetime(config['analysis_end_date'])

    # --- 3. 為替シミュレーション ---
    print("Running FX Simulation...")
    fx_scenarios = run_fx_simulation(
        historical_fx_df, sim_start, sim_end, 
        method=config['simulation_method'], 
        n_sims=config['n_simulations']
    )
    
    # --- 4. 収益計算 ---
    profits, details_df = calculate_profits(contracts_df, fx_scenarios, config)
    
    # --- 5. 集計 ---
    expected_profit = np.mean(profits)
    p5 = np.percentile(profits, 5)
    p95 = np.percentile(profits, 95)
    ko_probs = calc_ko_probabilities(contracts_df, details_df, config['n_simulations'])
    
    print("\n" + "="*40)
    print(f"Expected Profit: {expected_profit:,.0f}")
    print(f"Range (P5-P95):  {p5:,.0f} - {p95:,.0f}")
    print("="*40)
    
    # --- 6. 可視化 ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # A. 収益分布
    sns.histplot(profits, kde=True, ax=axes[0,0])
    axes[0,0].axvline(expected_profit, color='r', linestyle='--')
    axes[0,0].set_title('Profit Distribution')
    axes[0,0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # B. 消滅確率（上位）
    if not ko_probs.empty:
        ko_probs.head(20).plot(kind='barh', ax=axes[0,1])
        axes[0,1].invert_yaxis()
        axes[0,1].set_title('Top KO Probabilities')
    
    # C. 為替パス（USDJPY例）
    if 'USDJPY' in historical_fx_df.columns:
        # 実績
        hist_plot = historical_fx_df['USDJPY'].loc['2025-01-01':]
        hist_plot.plot(ax=axes[1,0], color='black', label='Historical')
        # シミュレーション（先頭20本）
        for sc in fx_scenarios[:20]:
            sc['USDJPY'].loc[sim_start:].plot(ax=axes[1,0], color='grey', alpha=0.3, legend=False)
        axes[1,0].set_title('USDJPY Paths')
        
    # D. 月次期待収益
    if not details_df.empty:
        # シナリオ毎・月毎に集計して平均
        monthly = details_df.groupby(['Scenario', pd.Grouper(key='Date', freq='M')])['Profit'].sum().reset_index()
        avg_monthly = monthly.groupby('Date')['Profit'].mean()
        avg_monthly.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Monthly Expected Profit')
        axes[1,1].set_xticklabels([d.strftime('%Y-%m') for d in avg_monthly.index], rotation=45)
        axes[1,1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        
    plt.tight_layout()
    plt.show()

    # 詳細確認用（KO確率詳細）
    if not ko_probs.empty:
        print("\nTop KO Contracts Details:")
        top_ids = ko_probs.head(5).index
        cols = ['契約ID', '_Type', '受取通貨', 'KO・KI相場(1行目)', 'ターゲット金額（円）']
        print(contracts_df[contracts_df['契約ID'].isin(top_ids)][cols])

