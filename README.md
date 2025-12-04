import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.ticker as mticker

# グラフの日本語表示設定（必要に応じてコメントアウトまたはフォント名を変更してください）
# try:
#     # 例: Windowsの場合 'MS Gothic' や 'Meiryo', Mac/Linuxの場合 'IPAexGothic' など
#     plt.rcParams['font.family'] = 'MS Gothic' 
# except:
#     print("Info: 日本語フォントの設定はスキップされました。グラフが文字化けする可能性があります。")

# 再現性確保のため乱数シードを固定
np.random.seed(42)

# --------------------------------------------------------------------------------
# 1. ダミーデータの生成（または実データのロード）
# --------------------------------------------------------------------------------

def generate_dummy_data(n_contracts=100):
    """
    ダミーの契約データと為替レート実績データを生成する。
    【重要】実データを使用する場合は、ここでファイルをロードし、DataFrameを返してください。
    カラム名は本コードで使用しているものと一致させてください。
    """

    # 為替レート実績データ (historical_fx_df)
    start_date = datetime(2023, 1, 1)
    # 現在の日付（コンテキストに合わせて2025/12/3と仮定）
    today = datetime(2025, 12, 3)
    end_date = today
    # 営業日ベースで生成
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)

    def generate_fx_series(start_price, volatility, n_days, seed):
        rng = np.random.default_rng(seed=seed)
        # 日次リターンを生成
        returns = rng.normal(0, volatility / np.sqrt(252), n_days)
        # 累積リターンから価格パスを生成
        prices = start_price * np.exp(np.cumsum(returns))
        return prices

    historical_fx_df = pd.DataFrame({
        'Date': dates,
        'USDJPY': generate_fx_series(150, 0.10, n_days, seed=42),
        'EURJPY': generate_fx_series(160, 0.08, n_days, seed=43),
        'CNHJPY': generate_fx_series(20, 0.05, n_days, seed=44)
    }).set_index('Date')

    # 契約データ (contracts_df)
    contracts = []
    rng_c = np.random.default_rng(seed=42)

    for i in range(n_contracts):
        # 基本情報
        contract_date = start_date + timedelta(days=int(rng_c.integers(0, (today - start_date).days)))
        duration_years = rng_c.integers(1, 6)
        end_date_c = contract_date + relativedelta(years=duration_years)
        first_settlement_date = contract_date + relativedelta(months=1)

        # 通貨ペアと方向性
        if rng_c.random() > 0.5:
            # 輸入型 (外貨受取)
            receive_currency = rng_c.choice(['USD', 'EUR', 'CNH'])
            pay_currency = 'JPY'
            direction = 'Import'
        else:
            # 輸出型 (円受取)
            receive_currency = 'JPY'
            pay_currency = rng_c.choice(['USD', 'EUR', 'CNH'])
            direction = 'Export'

        # デリバティブの種類
        derivative_type = rng_c.choice(['NonKO', 'NormalKO', 'TargetKO_Amount', 'TargetKO_Count'], p=[0.4, 0.3, 0.2, 0.1])

        contract = {
            '契約ID': f'C{i+1:04d}',
            '契約日': contract_date,
            '終了日': end_date_c,
            '初回決済日': first_settlement_date,
            '消滅日': np.nan,
            '収益': rng_c.integers(100, 1000) * 10000,
            '受取通貨': receive_currency,
            '支払通貨': pay_currency,
            '通常KO型': 1 if derivative_type == 'NormalKO' else 0,
            'ターゲットKO型': 1 if 'TargetKO' in derivative_type else 0,
            'KO・KI相場(1行目)': np.nan,
            'KO・KI判定(1行目)': np.nan,
            '年間交換回数': rng_c.choice([1, 2, 4, 12]),
            '交換日': rng_c.choice([15, 25, '末']),
            '消滅判定開始日（ウィンドウ）': np.nan,
            'ターゲット金額（円）': np.nan,
            'ターゲット回数（件数）': np.nan,
            '決済相場1': np.nan,
            '決済相場2': np.nan,
            '受取': rng_c.integers(1, 100) * 10000,
            '支払件数(特殊形の抽出後)': np.nan,
            '_Type': derivative_type # 分析用補助カラム
        }

        # 種類に応じたパラメータ設定
        base_rate = 150 if 'USD' in [receive_currency, pay_currency] else (160 if 'EUR' in [receive_currency, pay_currency] else 20)

        if derivative_type == 'NormalKO':
            if direction == 'Import':
                contract['KO・KI相場(1行目)'] = base_rate * (1 + rng_c.uniform(0.05, 0.20))
            else:
                contract['KO・KI相場(1行目)'] = base_rate * (1 - rng_c.uniform(0.05, 0.20))
            contract['KO・KI判定(1行目)'] = rng_c.choice(['ｱﾒﾘｶﾝ', 'ﾖｰﾛﾋﾟｱﾝ'])

        elif 'TargetKO' in derivative_type:
            # ターゲットKOは「ターゲットレートをシミュレーションレートが上回った」場合に蓄積
            contract['決済相場1'] = base_rate * rng_c.uniform(0.90, 1.10)
            # 2段階設定
            if rng_c.random() > 0.7:
                contract['決済相場2'] = base_rate * rng_c.uniform(0.90, 1.10)
                contract['支払件数(特殊形の抽出後)'] = rng_c.integers(3, 12)

            if derivative_type == 'TargetKO_Amount':
                contract['ターゲット金額（円）'] = contract['受取'] * rng_c.uniform(5, 25)
            elif derivative_type == 'TargetKO_Count':
                contract['ターゲット回数（件数）'] = rng_c.integers(5, 20)

        # ウィンドウ設定 (10%の確率で設定)
        if rng_c.random() < 0.1:
            window_start = first_settlement_date + relativedelta(months=int(rng_c.integers(1, 12)))
            if window_start < end_date_c:
                 contract['消滅判定開始日（ウィンドウ）'] = window_start

        # 過去の消滅確定 (10%の確率で設定)
        if rng_c.random() < 0.1 and contract['契約日'] < (today - relativedelta(months=3)):
             # 初回決済日から今日までの期間でランダムに消滅日を設定
             if first_settlement_date < today:
                 possible_ko_days = (min(end_date_c, today) - first_settlement_date).days
                 if possible_ko_days > 0:
                    ko_date = first_settlement_date + timedelta(days=int(rng_c.integers(0, possible_ko_days)))
                    contract['消滅日'] = ko_date

        contracts.append(contract)

    contracts_df = pd.DataFrame(contracts)
    # 日付型の変換
    date_cols = ['契約日', '終了日', '初回決済日', '消滅日', '消滅判定開始日（ウィンドウ）']
    for col in date_cols:
        contracts_df[col] = pd.to_datetime(contracts_df[col])

    return contracts_df, historical_fx_df

# --------------------------------------------------------------------------------
# 2. 為替レートシミュレーション
# --------------------------------------------------------------------------------

class FXSimulator:
    """為替レートのシミュレーションを行うクラス"""
    def __init__(self, historical_fx_df, simulation_start_date, simulation_end_date, method='GBM', target_rates=None):
        self.historical_fx_df = historical_fx_df
        self.simulation_start_date = pd.to_datetime(simulation_start_date)
        self.simulation_end_date = pd.to_datetime(simulation_end_date)
        self.method = method
        self.target_rates = target_rates
        self.currencies = [col for col in historical_fx_df.columns if 'JPY' in col]
        self.params = self._estimate_parameters()

    def _estimate_parameters(self):
        """過去データからドリフト(mu)とボラティリティ(sigma)を推定する (GBM用)"""
        params = {}
        for currency in self.currencies:
            if currency in self.historical_fx_df.columns:
                # 対数リターンを計算
                log_returns = np.log(self.historical_fx_df[currency] / self.historical_fx_df[currency].shift(1)).dropna()
                # 年率換算 (営業日数を252日と仮定)
                mu = log_returns.mean() * 252
                sigma = log_returns.std() * np.sqrt(252)
                params[currency] = {'mu': mu, 'sigma': sigma}
        return params

    def simulate(self, n_simulations, seed=42):
        """シミュレーションを実行し、結合されたレートデータ（実績+将来）を返す"""
        simulation_dates = pd.date_range(start=self.simulation_start_date, end=self.simulation_end_date, freq='B')
        n_days = len(simulation_dates)
        
        # シミュレーション結果を格納するリスト。各要素が1つのシナリオ（実績+将来の完全なDataFrame）
        all_simulations = []
        
        # 乱数生成器の初期化
        rng = np.random.default_rng(seed=seed)

        # シミュレーション開始直前の最終価格を取得
        last_prices = {}
        # simulation_start_dateの前日までの最新レートを取得
        last_historical_date = self.simulation_start_date - timedelta(days=1)
        for currency in self.currencies:
             if currency in self.historical_fx_df.columns:
                # asofを使用して直近の過去の営業日のレートを取得
                last_prices[currency] = self.historical_fx_df[currency].asof(last_historical_date)

        for i in range(n_simulations):
            simulated_data = {}
            for currency in self.currencies:
                if currency not in last_prices or pd.isna(last_prices[currency]):
                    continue

                last_price = last_prices[currency]

                if self.method == 'GBM':
                    mu = self.params[currency]['mu']
                    sigma = self.params[currency]['sigma']
                    dt = 1/252
                    # GBMのドリフト項 (μ - 0.5*σ^2)dt
                    drift = (mu - 0.5 * sigma**2) * dt
                    
                    # 乱数生成
                    random_shocks = rng.normal(0, 1, n_days)
                    # 拡散項 σ * sqrt(dt) * Z
                    diffusion = sigma * np.sqrt(dt) * random_shocks
                    
                    log_returns = drift + diffusion
                    # 価格パスの計算 S_t = S_0 * exp(cumsum(log_returns))
                    prices = last_price * np.exp(np.cumsum(log_returns))

                elif self.method == 'Linear':
                    # 線形モデル (Linearモデルの場合、n_simulationsは通常1)
                    if self.target_rates and currency in self.target_rates:
                        target_price = self.target_rates[currency]
                    else:
                        # ターゲット指定がない場合は最終価格維持
                        target_price = last_price
                    # 開始価格から目標価格まで線形補間
                    prices = np.linspace(last_price, target_price, n_days+1)[1:]

                else:
                    raise ValueError(f"Unsupported simulation method: {self.method}")

                simulated_data[currency] = prices

            simulated_df = pd.DataFrame(simulated_data, index=simulation_dates)

            # 実績データとシミュレーションデータを結合
            combined_df = pd.concat([self.historical_fx_df, simulated_df]).sort_index()
            # 欠損値を前方補完（土日祝日などでレートがない場合の対処）。重要。
            combined_df = combined_df.ffill()
            all_simulations.append(combined_df)

        return all_simulations

# --------------------------------------------------------------------------------
# 3. ユーティリティ関数
# --------------------------------------------------------------------------------

def get_exchange_dates(start_date, end_date, annual_frequency, exchange_day):
    """
    交換日のリストを計算する。
    [初回決済日](start_date)を起点とし、[年間交換回数]と[交換日]に基づき日付リストを生成する。
    """
    if annual_frequency <= 0 or pd.isna(annual_frequency) or pd.isna(start_date) or pd.isna(end_date) or pd.isna(exchange_day):
        return []

    interval_months = 12 // annual_frequency
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    dates = []
    # イテレーションの開始点は初回決済日の月の初日とする
    current_date = start_date.replace(day=1)

    while current_date <= end_date:
        if exchange_day == '末':
            # relativedelta(day=31) は自動的にその月の最終日になる
            target_date = current_date + relativedelta(day=31)
        else:
            try:
                day = int(exchange_day)
                # relativedelta(day=day) は指定日がその月に存在しない場合、月末に調整される
                target_date = current_date + relativedelta(day=day)
            except (ValueError, TypeError):
                # 数値変換できない場合はスキップ
                current_date += relativedelta(months=interval_months)
                continue

        # 計算された交換日が初回決済日以降かつ終了日以前であれば追加
        if target_date >= start_date and target_date <= end_date:
            dates.append(target_date)

        # 次の交換月へ移動
        current_date += relativedelta(months=interval_months)

    return sorted(list(set(dates)))


def get_fx_rate(fx_df, date, currency_pair):
    """指定された日付の為替レートを取得する（直近の過去のレートを取得）"""
    if currency_pair not in fx_df.columns:
        return np.nan
    try:
        # asofは指定された日付またはそれ以前の最新のレートを返す (ffillされたデータに対して有効)
        return fx_df[currency_pair].asof(date)
    except KeyError:
        # 日付がインデックスの範囲外の場合
        return np.nan

# --------------------------------------------------------------------------------
# 4. 消滅判定ロジック
# --------------------------------------------------------------------------------

class KnockOutEvaluator:
    """各契約の消滅判定を行うクラス"""
    def __init__(self, contract, fx_scenario_df):
        self.contract = contract
        self.fx_df = fx_scenario_df
        self.receive_currency = contract['受取通貨']
        self.pay_currency = contract['支払通貨']
        # 輸入型（[受取通貨]がJPY以外）
        self.is_import = self.receive_currency != 'JPY'
        self.currency_pair = self._get_currency_pair()
        self.ko_date = None

    def _get_currency_pair(self):
        if 'USD' in [self.receive_currency, self.pay_currency]:
            return 'USDJPY'
        elif 'EUR' in [self.receive_currency, self.pay_currency]:
            return 'EURJPY'
        elif 'CNH' in [self.receive_currency, self.pay_currency]:
            return 'CNHJPY'
        return None

    def evaluate(self):
        """消滅判定を実行する"""
        # すでに消滅が確定している場合は判定しない
        if pd.notna(self.contract['消滅日']):
            return self.contract['消滅日']

        if self.currency_pair is None or self.currency_pair not in self.fx_df.columns:
             return None

        # 判定開始日の決定
        first_settlement_date = self.contract['初回決済日']
        window_start_date = self.contract['消滅判定開始日（ウィンドウ）']
        end_date = self.contract['終了日']

        if pd.isna(first_settlement_date) or pd.isna(end_date):
            return None

        # 要件：[初回決済日]以降から開始。ただし、[消滅判定開始日（ウィンドウ）]がある場合は、このカラムの日付から開始。
        # 安全のため、初回決済日とウィンドウ開始日の遅い方を採用する。
        if pd.notna(window_start_date):
            start_date_for_judgement = max(first_settlement_date, window_start_date)
        else:
            start_date_for_judgement = first_settlement_date
        
        if start_date_for_judgement > end_date:
            return None

        # 契約種別に応じた判定ロジックの呼び出し
        if self.contract['通常KO型'] == 1:
            self._evaluate_normal_ko(start_date_for_judgement, end_date, first_settlement_date)
        elif self.contract['ターゲットKO型'] == 1:
            self._evaluate_target_ko(start_date_for_judgement, end_date, first_settlement_date)

        return self.ko_date

    def _check_ko_condition(self, rate, ko_rate):
        """通常KO型のノックアウト条件を判定する"""
        if self.is_import:
            # 輸入型：ノックアウトレートよりも想定レートが上回ったら消滅
            return rate >= ko_rate
        else:
            # 輸出型：下回ったら消滅
            return rate <= ko_rate

    def _evaluate_normal_ko(self, start_date, end_date, first_settlement_date):
        """①通常KO型の判定"""
        ko_rate = self.contract['KO・KI相場(1行目)']
        if pd.isna(ko_rate):
            return

        judgement_type = self.contract['KO・KI判定(1行目)']

        if judgement_type == 'ｱﾒﾘｶﾝ':
            # アメリカンタイプ：日次で判定
            # 判定期間のレートを取得 (start_date以降)
            # locでスライスする際は、インデックスがソートされている必要がある
            if not self.fx_df.index.is_monotonic_increasing:
                self.fx_df = self.fx_df.sort_index()
                
            relevant_rates = self.fx_df[self.currency_pair].loc[start_date:end_date]
            
            if relevant_rates.empty:
                return

            ko_mask = self._check_ko_condition(relevant_rates, ko_rate)
            if ko_mask.any():
                # 最初に条件を満たした日を消滅日とする
                self.ko_date = ko_mask[ko_mask].index[0]

        elif judgement_type == 'ﾖｰﾛﾋﾟｱﾝ':
            # ヨーロピアンタイプ：交換日のみで判定
            # 交換日の計算基準は[初回決済日]
            exchange_dates = get_exchange_dates(
                first_settlement_date,
                end_date,
                self.contract['年間交換回数'],
                self.contract['交換日']
            )
            # 判定開始日(start_date)以降の交換日のみ対象
            exchange_dates = [d for d in exchange_dates if d >= start_date]

            for date in exchange_dates:
                rate = get_fx_rate(self.fx_df, date, self.currency_pair)
                if pd.notna(rate) and self._check_ko_condition(rate, ko_rate):
                    self.ko_date = date
                    break

    def _evaluate_target_ko(self, start_date, end_date, first_settlement_date):
        """②ターゲットKO型（金額）および ③ターゲットKO型（件数）の判定"""
        is_amount_target = pd.notna(self.contract['ターゲット金額（円）'])
        is_count_target = pd.notna(self.contract['ターゲット回数（件数）'])

        if not is_amount_target and not is_count_target:
            return

        target_rate1 = self.contract['決済相場1']
        target_rate2 = self.contract['決済相場2']
        switch_count = self.contract['支払件数(特殊形の抽出後)']
        amount_per_exchange = self.contract['受取']

        # 交換日の計算（初回決済日から計算）
        exchange_dates = get_exchange_dates(
            first_settlement_date,
            end_date,
            self.contract['年間交換回数'],
            self.contract['交換日']
        )

        exchange_count = 0
        accumulated_value = 0

        # 実績期間を含む全交換日をループし、蓄積値と交換回数を計算する
        for date in exchange_dates:
            exchange_count += 1

            # 判定開始日より前はスキップ（ただし交換回数はカウントする）
            if date < start_date:
                continue

            # ターゲットレートの決定（2段階切り替えロジック）
            current_target_rate = target_rate1
            if pd.notna(target_rate2) and pd.notna(switch_count) and exchange_count > switch_count:
                current_target_rate = target_rate2

            if pd.isna(current_target_rate):
                 continue

            # 為替レートの取得（実績またはシミュレーション）
            rate = get_fx_rate(self.fx_df, date, self.currency_pair)
            if pd.isna(rate):
                continue

            # 蓄積ロジック
            # 要件定義：「ターゲットレートをシミュレーションレートが上回った」場合に蓄積
            if rate > current_target_rate:
                if is_amount_target:
                    # ②金額ターゲット
                    profit = (rate - current_target_rate) * amount_per_exchange
                    accumulated_value += profit
                elif is_count_target:
                    # ③回数ターゲット
                    accumulated_value += 1
            # 下回る場合はマイナス値の加算などはしない（スキップまたはゼロを足す操作を行う）

            # 消滅判定
            target_threshold = self.contract['ターゲット金額（円）'] if is_amount_target else self.contract['ターゲット回数（件数）']
            
            if pd.notna(target_threshold) and accumulated_value >= target_threshold:
                self.ko_date = date
                break

# --------------------------------------------------------------------------------
# 5. シミュレーション実行と収益計算
# --------------------------------------------------------------------------------

class ProfitSimulator:
    """全体のシミュレーションと収益計算を統括するクラス"""
    def __init__(self, contracts_df, historical_fx_df, analysis_start_date, analysis_end_date,
                 roll_rate_non_ko=0.6, roll_rate_ko=0.7, simulation_method='GBM', target_rates=None, n_simulations=100):
        self.contracts_df = contracts_df.copy()
        self.historical_fx_df = historical_fx_df
        self.analysis_start_date = pd.to_datetime(analysis_start_date)
        self.analysis_end_date = pd.to_datetime(analysis_end_date)
        self.roll_rate_non_ko = roll_rate_non_ko
        self.roll_rate_ko = roll_rate_ko
        self.simulation_method = simulation_method
        
        # Linearモデルの場合はシミュレーション回数を1に強制
        if self.simulation_method == 'Linear':
            self.n_simulations = 1
        else:
            self.n_simulations = n_simulations

        # シミュレーション期間の設定
        # 実績データの最終日の翌日からシミュレーション開始
        self.simulation_start_date = historical_fx_df.index.max() + timedelta(days=1)

        if self.simulation_start_date > self.analysis_end_date:
             self.simulation_end_date = self.simulation_start_date
             print("Info: 分析期間がすべて過去のため、為替シミュレーションは実行されません。")
        else:
             # シミュレーション期間は分析終了日までとする
             self.simulation_end_date = self.analysis_end_date

        self.fx_simulator = FXSimulator(
            historical_fx_df,
            self.simulation_start_date,
            self.simulation_end_date,
            method=simulation_method,
            target_rates=target_rates
        )
        self.simulation_results = []
        self.ko_probabilities = None
        self.fx_scenarios = []

    def run_simulation(self, seed=42):
        """モンテカルロシミュレーションを実行する"""
        print(f"Starting simulation: Method={self.simulation_method}, Scenarios={self.n_simulations}")
        
        # 為替シナリオの生成
        # シミュレーション期間がある場合のみ実行
        if self.simulation_start_date <= self.simulation_end_date:
             self.fx_scenarios = self.fx_simulator.simulate(self.n_simulations, seed=seed)
        else:
             # シミュレーション不要の場合は、実績データのみを使用 (ffillで休日補完)
             self.fx_scenarios = [self.historical_fx_df.ffill()] * self.n_simulations

        print(f"FX simulation finished. Starting profit calculation...")

        # 各シナリオでの収益計算
        for i, fx_scenario_df in enumerate(self.fx_scenarios):
            # 進捗表示
            if self.n_simulations > 1 and ((i+1) % 100 == 0 or (i+1) == self.n_simulations):
                print(f"Processing scenario {i+1}/{self.n_simulations}")
                
            scenario_result = self._calculate_scenario_profit(fx_scenario_df)
            self.simulation_results.append(scenario_result)

        print("Profit calculation finished.")
        self._calculate_ko_probabilities()
        return self.get_summary()

    def _calculate_scenario_profit(self, fx_scenario_df):
        """1つのシナリオにおける収益を計算する"""
        total_profit = 0
        details = []

        for index, contract in self.contracts_df.iterrows():
            contract_id = contract['契約ID']
            original_profit = contract['収益']
            
            if pd.isna(original_profit):
                continue

            # 1. 期間中の契約収益（新規契約分）
            # 要件：「対象分析開始時点を外部設定で与え、まずこの期間に契約した収益を累積します。」
            if self.analysis_start_date <= contract['契約日'] <= self.analysis_end_date:
                total_profit += original_profit
                details.append({'契約ID': contract_id, 'Type': 'New Contract', 'Profit': original_profit, 'Date': contract['契約日']})

            # 2. ロール収益の計算

            # 2.1 消滅確定済みの契約
            # 要件：「期間中に消滅確定済みのものは、消滅日にロール率分だけロールした前提で収益を計上します。」
            if pd.notna(contract['消滅日']):
                if self.analysis_start_date <= contract['消滅日'] <= self.analysis_end_date:
                    # 消滅したものはKO扱いのロール率(0.7)を使用
                    roll_profit = original_profit * self.roll_rate_ko
                    total_profit += roll_profit
                    details.append({'契約ID': contract_id, 'Type': 'Roll (Historical KO)', 'Profit': roll_profit, 'Date': contract['消滅日']})
                continue

            # 2.2 消滅条件なしの契約
            is_non_ko = (contract['通常KO型'] == 0) and (contract['ターゲットKO型'] == 0)
            if is_non_ko:
                # 要件：「満期が訪れるか否かのみを判定し、訪れる場合のみロール収益を計上します。」(ロール率0.6)
                if self.analysis_start_date <= contract['終了日'] <= self.analysis_end_date:
                    roll_profit = original_profit * self.roll_rate_non_ko
                    total_profit += roll_profit
                    details.append({'契約ID': contract_id, 'Type': 'Roll (Maturity)', 'Profit': roll_profit, 'Date': contract['終了日']})
                continue

            # 2.3 消滅条件ありの契約（シミュレーション）
            evaluator = KnockOutEvaluator(contract, fx_scenario_df)
            ko_date_simulated = evaluator.evaluate()

            if ko_date_simulated:
                # シミュレーションで消滅した場合
                # 要件：「消滅するものについては、ロール率0.7（可変設定2）をかけて収益を求めに行く」
                if self.analysis_start_date <= ko_date_simulated <= self.analysis_end_date:
                    roll_profit = original_profit * self.roll_rate_ko
                    total_profit += roll_profit
                    details.append({'契約ID': contract_id, 'Type': 'Roll (Simulated KO)', 'Profit': roll_profit, 'Date': ko_date_simulated})
            else:
                # シミュレーションで消滅しなかった場合（満期まで残存）
                # 満期が分析期間内に到来する場合のロール収益を計上（NonKOと同じ扱い0.6とするのが自然）
                 if self.analysis_start_date <= contract['終了日'] <= self.analysis_end_date:
                    roll_profit = original_profit * self.roll_rate_non_ko
                    total_profit += roll_profit
                    details.append({'契約ID': contract_id, 'Type': 'Roll (Simulated Maturity)', 'Profit': roll_profit, 'Date': contract['終了日']})

        return {
            'TotalProfit': total_profit,
            'Details': pd.DataFrame(details)
        }

    def _calculate_ko_probabilities(self):
        """各契約の消滅確率を計算する（消滅シナリオ数 ÷ シミュレーション全体）"""
        ko_counts = {}
        # シミュレーション対象となった契約（未消滅かつKO条件あり）を抽出
        target_contracts = self.contracts_df[
            (self.contracts_df['消滅日'].isna()) &
            ((self.contracts_df['通常KO型'] == 1) | (self.contracts_df['ターゲットKO型'] == 1))
        ]['契約ID']

        for contract_id in target_contracts:
            ko_count = 0
            for result in self.simulation_results:
                details = result['Details']
                # 当該契約がこのシナリオで「シミュレーションによりKO」したか（分析期間内にKOしたことを確認）
                is_ko = ((details['契約ID'] == contract_id) & (details['Type'] == 'Roll (Simulated KO)')).any()
                if is_ko:
                    ko_count += 1
            ko_counts[contract_id] = ko_count

        # 確率計算
        if self.n_simulations > 0:
            probabilities = pd.Series(ko_counts) / self.n_simulations
        else:
            probabilities = pd.Series(ko_counts, dtype=float)
            
        self.ko_probabilities = probabilities.sort_values(ascending=False)

    def get_summary(self):
        """シミュレーション結果の要約（期待値、上下限）を返す"""
        if not self.simulation_results:
            return None

        profits = [result['TotalProfit'] for result in self.simulation_results]
        profit_series = pd.Series(profits)

        summary = {
            'ExpectedProfit': profit_series.mean(),
            'Profit_P5': profit_series.quantile(0.05), # 下限 (5パーセンタイル)
            'Profit_P95': profit_series.quantile(0.95), # 上限 (95パーセンタイル)
            'ProfitDistribution': profits,
            'KOProbabilities': self.ko_probabilities
        }
        return summary

    def get_detailed_results(self):
        """全シナリオの詳細な結果を結合して返す"""
        all_details = []
        for i, result in enumerate(self.simulation_results):
            details = result['Details'].copy()
            details['Scenario'] = i + 1
            all_details.append(details)
        if not all_details:
            return pd.DataFrame()
        return pd.concat(all_details)

# --------------------------------------------------------------------------------
# 6. 実行と可視化
# --------------------------------------------------------------------------------

def run_analysis(contracts_df, historical_fx_df, analysis_config, seed=42):
    """設定に基づき分析を実行し、結果を可視化する"""

    # 分析の実行
    profit_simulator = ProfitSimulator(
        contracts_df=contracts_df,
        historical_fx_df=historical_fx_df,
        analysis_start_date=analysis_config['analysis_start_date'],
        analysis_end_date=analysis_config['analysis_end_date'],
        roll_rate_non_ko=analysis_config['roll_rate_non_ko'],
        roll_rate_ko=analysis_config['roll_rate_ko'],
        simulation_method=analysis_config['simulation_method'],
        target_rates=analysis_config.get('target_rates'),
        n_simulations=analysis_config['n_simulations']
    )

    summary = profit_simulator.run_simulation(seed=seed)

    if summary is None:
        print("Simulation failed or no results generated.")
        return None, None

    # 結果の表示
    print("\n" + "="*30 + " Simulation Summary " + "="*30)
    print(f"Analysis Period: {analysis_config['analysis_start_date']} to {analysis_config['analysis_end_date']}")
    print(f"Simulation Method: {profit_simulator.simulation_method} ({profit_simulator.n_simulations} simulations)")
    print(f"Expected Profit (期待収益): {summary['ExpectedProfit']:,.0f}")
    print(f"Profit Range (P5 - P95) (収益レンジ): {summary['Profit_P5']:,.0f} - {summary['Profit_P95']:,.0f}")
    print("="*80)

    # 可視化
    plot_results(summary, profit_simulator)

    # 詳細データ
    detailed_results = profit_simulator.get_detailed_results()

    return summary, detailed_results


def plot_results(summary, simulator):
    """結果をグラフで可視化する"""

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('デリバティブポートフォリオ収益シミュレーション結果', fontsize=16)

    # 1. 収益分布のヒストグラム
    ax = axes[0, 0]
    if summary['ProfitDistribution'] and len(summary['ProfitDistribution']) > 1:
        sns.histplot(summary['ProfitDistribution'], kde=True, ax=ax)
        ax.axvline(summary['ExpectedProfit'], color='r', linestyle='--', label=f'期待値: {summary["ExpectedProfit"]:,.0f}')
        ax.axvline(summary['Profit_P5'], color='g', linestyle=':', label=f'P5: {summary["Profit_P5"]:,.0f}')
        ax.axvline(summary['Profit_P95'], color='g', linestyle=':', label=f'P95: {summary["Profit_P95"]:,.0f}')
        ax.legend()
        ax.set_title('総収益の分布 (モンテカルロ)')
    else:
        # 単一シナリオの場合 (Linearモデルなど)
        ax.set_title('総収益 (単一シナリオ)')
        if summary['ProfitDistribution']:
            profit = summary['ProfitDistribution'][0]
            ax.bar(['収益'], [profit])
            ax.text(0, profit, f'{profit:,.0f}', ha='center', va='bottom')

    ax.set_xlabel('総収益')
    ax.set_ylabel('頻度')
    # X軸のフォーマットをカンマ区切りにする
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))


    # 2. 消滅確率の棒グラフ（上位20件）
    ax = axes[0, 1]
    # 消滅確率が0より大きいものの上位20件
    top_ko_probs = summary['KOProbabilities'][summary['KOProbabilities'] > 0].head(20)
    if not top_ko_probs.empty:
        top_ko_probs.plot(kind='barh', ax=ax)
        ax.set_title('消滅確率 上位20契約（分析期間内）')
        ax.set_xlabel('確率')
        ax.set_ylabel('契約ID')
        ax.invert_yaxis() # 降順表示
    else:
        ax.text(0.5, 0.5, 'シミュレーション上の消滅なし', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title('消滅確率')

    # 3. 為替レートシミュレーションのパス（例としてUSDJPY）
    ax = axes[1, 0]
    target_currency = 'USDJPY'
    if target_currency in simulator.fx_simulator.historical_fx_df.columns:
        # 過去データ（分析開始の約6ヶ月前から表示）
        start_date_plot = simulator.analysis_start_date - relativedelta(months=6)
        historical_data = simulator.fx_simulator.historical_fx_df[target_currency]
        
        if not historical_data.empty:
            # 過去データが存在する期間のみプロット
            historical_data = historical_data.loc[max(historical_data.index.min(), start_date_plot):]
            historical_data.plot(ax=ax, label='実績', color='black', linewidth=2)

        # シミュレーションパス（最初の20パスのみ表示）
        if simulator.simulation_start_date <= simulator.simulation_end_date:
            for i in range(min(len(simulator.fx_scenarios), 20)):
                scenario = simulator.fx_scenarios[i]
                sim_data = scenario[target_currency].loc[simulator.simulation_start_date:]
                sim_data.plot(ax=ax, color='grey', alpha=0.3, legend=False)

        ax.set_title(f'{target_currency} 為替レートシミュレーション (サンプルパス)')
        ax.set_xlabel('日付')
        ax.set_ylabel('レート')
        
        # 凡例の重複を避ける
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles=[handles[0]], labels=[labels[0]])
    else:
        ax.text(0.5, 0.5, f'{target_currency} データなし', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


    # 4. 月次収益の期待値推移
    ax = axes[1, 1]
    detailed_results = simulator.get_detailed_results()
    if not detailed_results.empty:
        # シナリオごとの月次収益を集計し、月ごとの平均収益（期待値）を計算
        monthly_profit = detailed_results.groupby(['Scenario', pd.Grouper(key='Date', freq='M')])['Profit'].sum().reset_index()
        expected_monthly_profit = monthly_profit.groupby('Date')['Profit'].mean()

        expected_monthly_profit.plot(kind='bar', ax=ax)
        ax.set_title('月次期待収益の推移')
        ax.set_xlabel('月')
        ax.set_ylabel('期待収益')
        if not expected_monthly_profit.empty:
            # X軸のラベルを年月表示にする
            ax.set_xticklabels([d.strftime('%Y-%m') for d in expected_monthly_profit.index], rotation=45, ha='right')
        # Y軸のフォーマットをカンマ区切りにする
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    else:
        ax.text(0.5, 0.5, '収益詳細データなし', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def analyze_details(summary, details, contracts_df):
    """シミュレーション結果の詳細分析（データ抽出用）"""
    if summary is None or details.empty:
        return None, None

    print("\n--- Detailed Analysis (詳細分析) ---")

    # 1. 収益タイプ別の期待値
    print("\n▼ Expected Profit by Type (収益タイプ別の期待値):")
    # シナリオごとの合計を計算し、その平均（期待値）を求める
    expected_profit_by_type = details.groupby(['Scenario', 'Type'])['Profit'].sum().groupby('Type').mean().sort_values(ascending=False)
    print(expected_profit_by_type.map('{:,.0f}'.format))

    # 2. 消滅確率の詳細データフレーム
    print("\n▼ KO Probabilities Details (消滅確率の詳細 - 確率>0のみ):")
    ko_probs = summary['KOProbabilities']
    ko_details = ko_probs[ko_probs > 0].to_frame(name='KO_Probability')
    
    # 契約情報を結合
    columns_to_join = ['_Type', '受取通貨', '支払通貨', '契約日', '終了日', '収益',
                       'KO・KI相場(1行目)', 'KO・KI判定(1行目)', 
                       '決済相場1', '決済相場2', 'ターゲット金額（円）', 'ターゲット回数（件数）']
    
    ko_details = ko_details.join(contracts_df.set_index('契約ID')[columns_to_join])
    
    # Jupyter Notebook環境で見やすく表示する場合は print の代わりに display() を使用
    # display(ko_details)
    print(ko_details)
    
    return expected_profit_by_type, ko_details

# --------------------------------------------------------------------------------
# 実行例（Jupyterでこのブロックを実行してください）
# --------------------------------------------------------------------------------

if __name__ == '__main__':
    # 1. データの準備（ダミーデータ生成または実データロード）
    print("Preparing data...")
    # ダミーデータの契約数を設定して生成
    contracts_df, historical_fx_df = generate_dummy_data(n_contracts=500) 

    # 2. 分析設定
    # ※分析対象期間やパラメータはここで設定します。

    # 例：今年度・来年度（2025年度・2026年度）の分析設定 (GBMモデル)
    # 期間: 2025/4/1 から 2027/3/31
    analysis_config_gbm = {
        'analysis_start_date': '2025-04-01',
        'analysis_end_date': '2027-03-31',
        'roll_rate_non_ko': 0.6, # 可変設定1
        'roll_rate_ko': 0.7,     # 可変設定2
        'simulation_method': 'GBM',
        'n_simulations': 1000    # シミュレーション回数
    }

    # 例：線形モデルでの分析設定（特定の円安シナリオ）
    analysis_config_linear = {
        'analysis_start_date': '2025-04-01',
        'analysis_end_date': '2027-03-31',
        'roll_rate_non_ko': 0.6,
        'roll_rate_ko': 0.7,
        'simulation_method': 'Linear',
        # 到達点のレートを設定
        'target_rates': {'USDJPY': 170.0, 'EURJPY': 185.0, 'CNHJPY': 25.0},
        'n_simulations': 1 # 線形モデルは1回
    }

    # 3. 分析の実行
    
    # --- GBMモデルによるリスク評価 ---
    print("\n" + "="*60)
    print("【実行例1】 GBMモデルによる分析 (リスク評価)")
    print("="*60)
    # 再現性確保のためseedを指定
    summary_gbm, details_gbm = run_analysis(contracts_df, historical_fx_df, analysis_config_gbm, seed=42)

    # 4. 追加分析（GBMモデル）
    if summary_gbm:
        expected_profit_gbm, ko_details_gbm = analyze_details(summary_gbm, details_gbm, contracts_df)
        # 結果をCSVで保存する場合の例
        # ko_details_gbm.to_csv('ko_probabilities_gbm.csv', encoding='utf-8-sig')


    # # --- 線形モデルによるシナリオ分析 ---
    # print("\n" + "="*60)
    # print("【実行例2】 線形モデルによる分析 (シナリオ分析)")
    # print("="*60)
    # summary_linear, details_linear = run_analysis(contracts_df, historical_fx_df, analysis_config_linear, seed=42)

    # # 5. 追加分析（線形モデル）
    # if summary_linear:
    #      analyze_details(summary_linear, details_linear, contracts_df)
