import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# 日本語フォント設定（環境に合わせて変更してください。Windowsなら'Meiryo'など）
plt.rcParams['font.family'] = 'sans-serif' 
# plt.rcParams['font.family'] = 'Meiryo' # Windowsの場合の例

class DerivativeSimulator:
    def __init__(self, df_contracts, history_rates, analysis_start, analysis_end):
        """
        df_contracts: 契約データ
        history_rates: 通貨ごとの過去実績レート (DataFrame index=Date, columns=['USD', 'EUR', ...])
        analysis_start: 分析開始日 (この日以降の収益を計算)
        analysis_end: 分析終了日
        """
        self.df = df_contracts.copy()
        self.history = history_rates
        self.start_date = pd.to_datetime(analysis_start)
        self.end_date = pd.to_datetime(analysis_end)
        
        # 設定パラメータ
        self.roll_rate_ko = 0.7  # KO消滅時のロール率
        self.roll_rate_maturity = 0.6  # 満期時のロール率
        
    def generate_scenarios_gbm(self, currency, num_scenarios, days, drift=0.0, vol=0.1):
        """
        幾何ブラウン運動によるモンテカルロシミュレーション
        """
        last_rate = self.history[currency].iloc[-1]
        dt = 1/252
        
        # ランダムウォーク生成
        shock = np.random.normal(0, 1, (days, num_scenarios))
        prices = np.zeros((days + 1, num_scenarios))
        prices[0] = last_rate
        
        for t in range(1, days + 1):
            prices[t] = prices[t-1] * np.exp((drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * shock[t-1])
            
        # 日付インデックスの作成
        start_sim_date = self.history.index[-1] + timedelta(days=1)
        date_index = pd.date_range(start=start_sim_date, periods=days+1, freq='B') # 営業日ベース
        
        return pd.DataFrame(prices, index=date_index)

    def generate_scenarios_linear(self, currency, target_rate, days):
        """
        現在値から目標値へ線形に動く単純シナリオ（1シナリオのみ）
        """
        last_rate = self.history[currency].iloc[-1]
        prices = np.linspace(last_rate, target_rate, days + 1)
        
        start_sim_date = self.history.index[-1] + timedelta(days=1)
        date_index = pd.date_range(start=start_sim_date, periods=days+1, freq='B')
        
        return pd.DataFrame(prices, index=date_index, columns=[0])

    def _get_exchange_dates(self, row, end_date):
        """
        初回決済日から終了日までの交換日リストを生成
        """
        dates = []
        current = pd.to_datetime(row['初回決済日'])
        maturity = pd.to_datetime(row['終了日'])
        
        # 交換頻度（月数）
        freq_months = int(12 / row['年間交換回数'])
        
        # 末日指定の処理
        is_month_end = str(row['交換日']) == '末'
        day_num = 31 if is_month_end else int(row['交換日'])
        
        while current <= maturity and current <= end_date:
            dates.append(current)
            # 次の予定日計算
            next_month = current + relativedelta(months=freq_months)
            if is_month_end:
                dates.append(next_month + relativedelta(day=31)) # その月の末日に補正
            else:
                try:
                    dates.append(next_month.replace(day=day_num))
                except ValueError:
                    # 2月30日などの場合、その月の末日にするなどの処理が必要だがここでは簡略化
                    dates.append(next_month + relativedelta(day=31))
            current = dates[-1]
            
        return sorted(list(set(dates))) # 重複排除してソート

    def check_ko(self, row, rate_path):
        """
        単一契約・単一シナリオに対する消滅判定
        戻り値: (is_ko, ko_date, ko_type)
        """
        # 通貨の判定
        ccy = row['受取通貨'] if row['受取通貨'] != 'JPY' else row['支払通貨']
        if ccy not in rate_path.columns:
            return False, None, None # 対象通貨のシナリオがない（金利系など）
            
        # パス結合（実績 + シナリオ）
        # ※本来はシナリオ開始日と契約期間のマージが必要ですが、ここでは簡略化して全期間参照可能とします
        full_path = pd.concat([self.history[ccy], rate_path[ccy]]).sort_index()
        full_path = full_path[~full_path.index.duplicated(keep='first')]

        # 期間外のデータを除外（初回決済日以降、かつウィンドウ開始日以降）
        start_check = pd.to_datetime(row['初回決済日'])
        if pd.notnull(row['消滅判定開始日（ウィンドウ）']):
            start_check = max(start_check, pd.to_datetime(row['消滅判定開始日（ウィンドウ）']))
            
        maturity = pd.to_datetime(row['終了日'])
        
        # 判定対象期間のデータ
        target_path = full_path[(full_path.index >= start_check) & (full_path.index <= maturity)]
        if target_path.empty:
            return False, None, None

        # 輸入/輸出判定 (JPY受取=輸出=下回ったらKO / JPY以外受取=輸入=上回ったらKO)
        is_export = (row['受取通貨'] == 'JPY')
        
        # --- ① 通常KO型 ---
        if row['通常KO型'] == 1:
            ko_rate = row['KO・KI相場(1行目)']
            is_american = (row['KO・KI判定(1行目)'] == 'ｱﾒﾘｶﾝ')
            
            if is_american:
                # 日次判定
                if is_export: # 下回ったら消滅
                    hit = target_path[target_path <= ko_rate]
                else: # 上回ったら消滅
                    hit = target_path[target_path >= ko_rate]
                
                if not hit.empty:
                    return True, hit.index[0], 'Normal_American'
            else:
                # ヨーロピアン（交換日のみ判定）
                ex_dates = self._get_exchange_dates(row, maturity)
                # target_pathに含まれる交換日のみ抽出
                valid_ex_dates = [d for d in ex_dates if d in target_path.index]
                
                for d in valid_ex_dates:
                    rate = target_path.loc[d]
                    is_hit = (rate <= ko_rate) if is_export else (rate >= ko_rate)
                    if is_hit:
                        return True, d, 'Normal_European'

        # --- ② & ③ ターゲットKO型 (累積ロジック) ---
        elif row['ターゲットKO型'] == 1:
            ex_dates = self._get_exchange_dates(row, maturity)
            valid_ex_dates = [d for d in ex_dates if d in target_path.index]
            
            accumulated_val = 0
            target_limit = row['ターゲット金額（円）'] if pd.notnull(row['ターゲット金額（円）']) else row['ターゲット回数（件数）']
            mode = 'amount' if pd.notnull(row['ターゲット金額（円）']) else 'count'
            
            # レート切り替え設定
            switch_count = row['支払件数(特殊形の抽出後)']
            rate1 = row['決済相場1']
            rate2 = row['決済相場2'] if pd.notnull(row['決済相場2']) else rate1
            
            for i, d in enumerate(valid_ex_dates):
                current_rate = target_path.loc[d]
                target_strike = rate1 if (i + 1) <= switch_count else rate2
                
                # 判定ロジック（輸入：Sim > Strike で蓄積 / 輸出：Sim < Strike で蓄積）
                # ※プロンプトの「上回った」等を基準に実装
                diff = 0
                if not is_export: # 輸入
                    if current_rate > target_strike:
                        diff = (current_rate - target_strike) if mode == 'amount' else 1
                else: # 輸出
                    if current_rate < target_strike:
                        diff = (target_strike - current_rate) if mode == 'amount' else 1 # 差分は絶対値的に扱うと仮定
                
                # 金額モードの場合、受取金額を掛ける
                if mode == 'amount' and diff > 0:
                    diff = diff * row['受取'] # ※受取カラムには1回あたりの外貨額が入っている前提
                
                accumulated_val += diff
                
                if accumulated_val >= target_limit:
                    return True, d, f'Target_{mode}'

        return False, None, None

    def run_simulation(self, scenarios_dict):
        """
        全契約・全シナリオを実行
        scenarios_dict: {'USD': df_scenarios, 'EUR': ...}
        """
        results = []
        
        # プログレスバーの代わりにプリント
        print(f"Simulation Start: {len(self.df)} Contracts")
        
        for idx, row in self.df.iterrows():
            # 通貨特定
            ccy = row['受取通貨'] if row['受取通貨'] != 'JPY' else row['支払通貨']
            
            # 金利系など対象外通貨はスキップまたは固定扱い
            if ccy not in scenarios_dict:
                # KO判定なしとして処理
                res = {
                    'contract_id': idx,
                    'revenue_base': row['収益'],
                    'revenue_roll': 0,
                    'is_ko': False,
                    'ko_date': None,
                    'scenario_id': 'Fixed'
                }
                # 満期ロール判定
                if pd.to_datetime(row['終了日']) <= self.end_date:
                     res['revenue_roll'] = row['収益'] * self.roll_rate_maturity
                results.append(res)
                continue

            scenarios = scenarios_dict[ccy]
            num_scenarios = scenarios.shape[1]
            
            # 既に消滅している場合
            if pd.notnull(row['消滅日']):
                ko_date = pd.to_datetime(row['消滅日'])
                roll_rev = 0
                # 分析期間内に消滅したならロール計上
                if self.start_date <= ko_date <= self.end_date:
                    roll_rev = row['収益'] * self.roll_rate_ko
                
                # シナリオ数分だけレコード複製（重み付けのため）
                for s in range(num_scenarios):
                    results.append({
                        'contract_id': idx,
                        'revenue_base': row['収益'], # 確定収益は常に計上（分析期間ロジックによるが、ここでは契約済み総収益として扱う）
                        'revenue_roll': roll_rev,
                        'is_ko': True,
                        'ko_date': ko_date,
                        'scenario_id': s,
                        'status': 'Already_Terminated'
                    })
                continue

            # シミュレーション実行
            for s_col in scenarios.columns:
                path = scenarios[[s_col]].rename(columns={s_col: ccy})
                is_ko, ko_date, ko_type = self.check_ko(row, path)
                
                rev_roll = 0
                status = 'Active'
                
                if is_ko:
                    if self.start_date <= ko_date <= self.end_date:
                        rev_roll = row['収益'] * self.roll_rate_ko
                    status = f'Simulated_KO ({ko_type})'
                else:
                    # 満期判定
                    maturity = pd.to_datetime(row['終了日'])
                    if self.start_date <= maturity <= self.end_date:
                        rev_roll = row['収益'] * self.roll_rate_maturity
                        status = 'Maturity'
                
                results.append({
                    'contract_id': idx,
                    'revenue_base': row['収益'],
                    'revenue_roll': rev_roll,
                    'is_ko': is_ko,
                    'ko_date': ko_date,
                    'scenario_id': s_col,
                    'status': status
                })
                
        return pd.DataFrame(results)

# --- 以下、実行用ダミーデータ生成とメイン処理 ---

def create_dummy_data(n=50):
    """分析に必要なカラムを持つダミーデータを作成"""
    data = []
    currencies = ['USD', 'EUR', 'CNH']
    types = ['Normal', 'Target_Amt', 'Target_Cnt']
    
    base_date = datetime(2020, 1, 1)
    
    for i in range(n):
        contract_date = base_date + timedelta(days=np.random.randint(0, 1000))
        maturity_years = np.random.choice([1, 2, 3, 5, 10])
        end_date = contract_date + relativedelta(years=maturity_years)
        
        ccy = np.random.choice(currencies)
        is_export = np.random.choice([True, False]) # True: JPY受取
        
        # 収益 (100万〜1000万)
        revenue = np.random.randint(100, 1000) * 10000
        
        row = {
            '契約日': contract_date,
            '終了日': end_date,
            '初回決済日': contract_date + relativedelta(months=1),
            '収益': revenue,
            '受取通貨': 'JPY' if is_export else ccy,
            '支払通貨': ccy if is_export else 'JPY',
            '年間交換回数': 12,
            '交換日': '末',
            '消滅日': pd.NaT, # 最初は生きている前提
            '消滅判定開始日（ウィンドウ）': pd.NaT,
            
            # フラグ初期化
            '通常KO型': 0,
            'KO・KI相場(1行目)': np.nan, 'KO・KI判定(1行目)': np.nan,
            'ターゲットKO型': 0, 'ターゲット金額（円）': np.nan, 'ターゲット回数（件数）': np.nan,
            '決済相場1': np.nan, '決済相場2': np.nan, '支払件数(特殊形の抽出後)': 0,
            '受取': 10000 # 1回あたりの外貨額（ダミー）
        }
        
        # タイプ別条件設定
        k_type = np.random.choice(types)
        current_spot = 110 if ccy=='USD' else 130 if ccy=='EUR' else 15
        
        if k_type == 'Normal':
            row['通常KO型'] = 1
            row['KO・KI判定(1行目)'] = np.random.choice(['ｱﾒﾘｶﾝ', 'ﾖｰﾛﾋﾟｱﾝ'])
            # 輸入なら上、輸出なら下にKOを設定
            barrier = current_spot * 1.1 if not is_export else current_spot * 0.9
            row['KO・KI相場(1行目)'] = barrier
            
        elif k_type == 'Target_Amt':
            row['ターゲットKO型'] = 1
            row['ターゲット金額（円）'] = revenue * 0.5 # ターゲット金額
            row['決済相場1'] = current_spot
            if np.random.random() > 0.5: # 変動ストライクあり
                row['決済相場2'] = current_spot * 1.05
                row['支払件数(特殊形の抽出後)'] = 12
                
        elif k_type == 'Target_Cnt':
            row['ターゲットKO型'] = 1
            row['ターゲット回数（件数）'] = 12 # 12回ヒットで終了
            row['決済相場1'] = current_spot
        
        # 一部を既に消滅させる
        if np.random.random() < 0.2:
            term_date = contract_date + timedelta(days=np.random.randint(10, 300))
            if term_date < datetime.now():
                row['消滅日'] = term_date

        data.append(row)
        
    return pd.DataFrame(data)

def create_dummy_history():
    """過去レートのダミー"""
    dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='B')
    df = pd.DataFrame(index=dates)
    # USD: 100 -> 150
    df['USD'] = np.linspace(100, 150, len(dates)) + np.random.normal(0, 1, len(dates))
    # EUR: 120 -> 160
    df['EUR'] = np.linspace(120, 160, len(dates)) + np.random.normal(0, 1, len(dates))
    # CNH: 15 -> 20
    df['CNH'] = np.linspace(15, 20, len(dates)) + np.random.normal(0, 0.1, len(dates))
    return df

# ==========================================
# メイン実行部
# ==========================================

# 1. データの準備
df_contracts = create_dummy_data(n=100) # 100件のダミー契約
history_rates = create_dummy_history()

# 分析期間設定（例：今年度と来年度）
today = datetime.now()
analysis_start = datetime(today.year, 4, 1) # 今年度4/1
analysis_end = datetime(today.year + 2, 3, 31) # 来年度末

print(f"分析期間: {analysis_start.strftime('%Y/%m/%d')} - {analysis_end.strftime('%Y/%m/%d')}")

# 2. シミュレーター初期化
sim = DerivativeSimulator(df_contracts, history_rates, analysis_start, analysis_end)

# 3. シナリオ生成 (モンテカルロ 100パス + 線形 1パス)
# 実際には通貨間の相関などを考慮する場合もありますが、今回は独立して生成
scenarios_dict = {}
sim_days = (analysis_end - today).days + 30 # 少し余裕を持たせる

for ccy in ['USD', 'EUR', 'CNH']:
    # モンテカルロ
    df_gbm = sim.generate_scenarios_gbm(ccy, num_scenarios=50, days=sim_days, drift=0.0, vol=0.15)
    df_gbm.columns = [f'Scen_{i}' for i in range(50)]
    
    # 線形（現状維持と円安進行の2パターン追加）
    current_val = history_rates[ccy].iloc[-1]
    df_flat = sim.generate_scenarios_linear(ccy, current_val, sim_days)
    df_depr = sim.generate_scenarios_linear(ccy, current_val * 1.2, sim_days) # 20%円安
    
    df_flat.columns = ['Linear_Flat']
    df_depr.columns = ['Linear_Depr']
    
    # 結合
    scenarios_dict[ccy] = pd.concat([df_gbm, df_flat, df_depr], axis=1)

# 4. シミュレーション実行
df_results = sim.run_simulation(scenarios_dict)

# ==========================================
# 集計と可視化
# ==========================================

# シナリオごとの総収益計算 (Base + Roll)
# ※Base収益は契約時に確定しているので、ここでの分析は「追加で得られるRoll収益の変動」が主眼になります
scenario_stats = df_results.groupby('scenario_id')[['revenue_base', 'revenue_roll']].sum()
scenario_stats['total_revenue'] = scenario_stats['revenue_base'] + scenario_stats['revenue_roll']

# 結果表示
print("\n--- シミュレーション結果概要 ---")
print(scenario_stats['total_revenue'].describe().apply(lambda x: f"{x:,.0f}"))

# グラフ化
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. 為替シナリオの可視化 (USDのみ例示)
ax = axes[0, 0]
scenarios_dict['USD'].plot(legend=False, alpha=0.3, color='blue', ax=ax)
scenarios_dict['USD'][['Linear_Flat', 'Linear_Depr']].plot(ax=ax, color='red', linewidth=2, label='Linear Scenarios')
ax.set_title('USD/JPY Simulation Paths')
ax.set_ylabel('Rate')
ax.grid(True)

# 2. 収益分布 (ヒストグラム)
ax = axes[0, 1]
sns.histplot(scenario_stats['total_revenue'] / 100000000, kde=True, ax=ax, color='green')
ax.set_title('Total Revenue Distribution (Simulation)')
ax.set_xlabel('Revenue (億円)')
ax.grid(True)
# 期待値ライン
exp_val = scenario_stats['total_revenue'].mean()
ax.axvline(exp_val / 100000000, color='red', linestyle='--', label=f'Expected: {exp_val/100000000:.2f}億')
ax.legend()

# 3. 契約ごとの消滅確率 (上位20件)
ax = axes[1, 0]
ko_counts = df_results[df_results['is_ko']].groupby('contract_id')['scenario_id'].count()
total_scenarios = len(scenarios_dict['USD'].columns)
ko_probs = (ko_counts / total_scenarios).sort_values(ascending=False).head(20)
ko_probs.plot(kind='bar', ax=ax, color='orange')
ax.set_title('KO Probability by Contract (Top 20)')
ax.set_ylabel('Probability')
ax.set_ylim(0, 1)
ax.grid(axis='y')

# 4. 消滅理由の内訳
ax = axes[1, 1]
status_counts = df_results['status'].value_counts()
status_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90)
ax.set_title('Outcome Breakdown (All Scenarios)')
ax.set_ylabel('')

plt.tight_layout()
plt.show()

# 追加分析用データの出力例
print("\n--- 詳細データ例 (df_results) ---")
print(df_results.head())

