import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
import hashlib
import json
import copy
import logging
import numpy as np

# ロギングの設定
logging.basicConfig(level=logging.WARNING)

# Streamlitページ設定
try:
    st.set_page_config(layout="wide", page_title="信連用 経営インパクトシミュレーター")
except st.errors.StreamlitAPIException:
    pass

# --- 定数定義 ---
NORINCHUKIN_MANDATORY_RATIO = 0.5
LCR_HQLA_HAIRCUT_AFS = 0.15
LCR_NET_OUTFLOW_RATIO = 0.05
# リスクテイク戦略のRWAへの影響度（利回り1%向上あたりのRW上昇幅）
RISK_TAKING_RWA_IMPACT_FACTOR = 50.0
# 浮動小数点誤差を考慮する閾値
TOLERANCE = 1e-9

# 色定義
COLOR_INCREASE = "#28a745"
COLOR_DECREASE = "#dc3545"
COLOR_PROFIT = "#4CAF50"
COLOR_JA_GREEN = "#006400"
COLOR_CAPITAL = "#FF9800"
COLOR_LIQUIDITY = "#2196F3"
COLOR_TARGET = "#6C757D"

# BSの項目名と色定義
BS_COLORS = {
    '必須預け金(農中)': '#388E3C',
    '任意預け金(農中)': '#66BB6A',
    '有価証券(HTM)': '#FFA726',
    '有価証券(AFS/その他)': '#FFC107',
    '貸出金': '#42A5F5',
    'その他資産': '#9E9E9E',
}

# --- サンプルデータの定義 ---
def get_sample_data_csv():
    """信連向けサンプルデータ（標準モデル）"""
    data = """
項目,金額（億円）
# 貸借対照表（資産）,総資産約6兆円規模のモデル
農林中金預け金（必須分）,27666
農林中金預け金（任意分）,2618
有価証券残高,13557
貸出金残高,11447
その他資産,5476
# 貸借対照表（負債・純資産）,
貯金残高,55332
# 現状の健全性指標,国内基準前提
自己資本（比率分子）,4484
その他有価証券評価差額金（AOCI）, -263
# 損益計算書（主要項目）,
経費（絶対額）,49
与信コスト（絶対額）,10
その他収支（絶対額）,52
# 主要な利回り・コスト（%）,奨励金等を含むネット値
農林中金預け金利回り（必須分）,0.724
農林中金預け金利回り（任意分）,0.10
有価証券利回り,1.02
貸出金利回り,0.51
貯金利回り（平均）,0.57
# その他パラメータ,
有価証券デュレーション,5.0
# リスクウェイト（信用リスクアセット計算用）
RW_預け金（%）,21.0
RW_有価証券（%）,85.0
RW_貸出金（%）,85.0
RW_その他資産（%）,56.94
固定RWA（億円）,2783
実効税率（%）,17.8
目標当期純利益,70.0
"""
    return data

# --- データロード関数 ---
@st.cache_data
def load_data(data_source):
    try:
        # 最初の2列のみを読み込む
        if isinstance(data_source, str):
            df = pd.read_csv(io.StringIO(data_source), comment='#', usecols=[0, 1])
        else:
            df = pd.read_csv(data_source, comment='#', usecols=[0, 1], encoding='utf-8-sig')
        df.columns = ['項目', '金額（億円）']
    except Exception as e:
        return None, f"CSVファイルの読み込みに失敗しました: {e}"

    data = {row['項目'].strip(): float(row['金額（億円）']) for index, row in df.iterrows() if pd.notna(row['項目']) and pd.notna(row['金額（億円）'])}

    initial_data = {}
    try:
        # BS項目 (億円 -> 円)。内部計算は円単位（Float）で行う。
        initial_data['BS'] = {
            'Deposits_JA': data['貯金残高'] * 1e8,
            'Securities_Total': data['有価証券残高'] * 1e8,
            'Loans': data['貸出金残高'] * 1e8,
            'Deposits_Norinchukin_Mandatory': data['農林中金預け金（必須分）'] * 1e8,
            'Deposits_Norinchukin_Voluntary': data['農林中金預け金（任意分）'] * 1e8,
            'OtherAssets': data.get('その他資産', 0) * 1e8,
            # 【修正】規制上の自己資本（国内基準分子）として明記
            'Equity_Regulatory': data['自己資本（比率分子）'] * 1e8,
            # その他有価証券評価差額金（税効果後）
            'AOCI': data.get('その他有価証券評価差額金（AOCI）', 0.0) * 1e8,
        }

        # PL項目
        initial_data['PL'] = {
            'Expenses': data['経費（絶対額）'] * 1e8,
            'CreditCost': data['与信コスト（絶対額）'] * 1e8,
            'OtherIncomeExpense': data.get('その他収支（絶対額）', 0) * 1e8,
            'TargetNetIncome': data.get('目標当期純利益', 0.0) * 1e8,
        }

        # パラメータ (% -> 比率)
        initial_data['Params'] = {
            'Rate_Norinchukin_Mandatory': data['農林中金預け金利回り（必須分）'] / 100,
            'Rate_Norinchukin_Voluntary': data['農林中金預け金利回り（任意分）'] / 100,
            'Rate_Securities': data['有価証券利回り'] / 100,
            'Rate_Loans': data['貸出金利回り'] / 100,
            'Rate_Deposits_Avg': data['貯金利回り（平均）'] / 100,
            'Securities_Duration': data['有価証券デュレーション'],
            'TaxRate': data['実効税率（%）'] / 100,
            'RW_Deposits': data['RW_預け金（%）'] / 100,
            'RW_Securities': data['RW_有価証券（%）'] / 100,
            'RW_Loans': data['RW_貸出金（%）'] / 100,
            'RW_OtherAssets': data.get('RW_その他資産（%）', 100.0) / 100,
            'FixedRWA': data['固定RWA（億円）'] * 1e8,
        }
    except KeyError as e:
        return None, f"必須項目が不足しています: {e}"

    return initial_data, None

# --- シミュレーションクラス ---
class ShinrenALMSimulator:
    def __init__(self, initial_data):
        self.initial_bs = copy.deepcopy(initial_data['BS'])
        self.initial_pl = copy.deepcopy(initial_data['PL'])
        self.params = copy.deepcopy(initial_data['Params'])
        self.base_case_results = self._calculate_base_case()
        self.initial_pl['NII'] = self.base_case_results['NII']
        self.initial_pl['OrdinaryProfit'] = self.base_case_results['OrdinaryProfit']
        self.initial_pl['NetIncome'] = self.base_case_results['NetIncome']

    def _get_securities_balance(self, bs):
        """有価証券残高（簿価）を取得する"""
        return bs.get('Securities_HTM', 0) + bs.get('Securities_AFS', 0) or bs.get('Securities_Total', 0)

    def _calculate_interest_income(self, bs, rates, dynamic_effects=0):
        securities_balance = self._get_securities_balance(bs)
        income = (
            bs['Deposits_Norinchukin_Mandatory'] * rates['Rate_Norinchukin_Mandatory'] +
            bs['Deposits_Norinchukin_Voluntary'] * rates['Rate_Norinchukin_Voluntary'] +
            securities_balance * rates['Rate_Securities'] +
            bs['Loans'] * rates['Rate_Loans'] +
            dynamic_effects
        )
        return income

    def _calculate_interest_expense(self, bs, rates):
        return bs['Deposits_JA'] * rates['Rate_Deposits_Avg']

    def _calculate_nii(self, bs, rates, dynamic_effects=0):
        return self._calculate_interest_income(bs, rates, dynamic_effects) - self._calculate_interest_expense(bs, rates)

    def _calculate_rwa(self, bs, rw_adjustment_factor=1.0):
        p = self.params
        cra_deposits = (bs['Deposits_Norinchukin_Mandatory'] + bs['Deposits_Norinchukin_Voluntary']) * p['RW_Deposits']

        adjusted_rw_securities = p['RW_Securities'] * rw_adjustment_factor
        cra_securities = self._get_securities_balance(bs) * adjusted_rw_securities

        cra_loans = bs['Loans'] * p['RW_Loans']
        cra_other = bs['OtherAssets'] * p['RW_OtherAssets']

        cra = cra_deposits + cra_securities + cra_loans + cra_other
        return cra + p['FixedRWA']

    def _calculate_liquidity(self, bs):
        """
        与えられたBS（AOCIを含む）からLCRを計算する。HQLAは時価ベースで評価。
        """
        # Level 1資産（任意預け金）。その他資産の算入（旧ロジック: OtherAssets * 0.1）は保守的に除外。
        hqla_level1 = bs.get('Deposits_Norinchukin_Voluntary', 0)

        # AFS残高（簿価）の特定
        afs_balance = bs.get('Securities_AFS', 0)
        # ベースケース（HTM/AFS区分がない場合）はSecurities_TotalをAFSとみなす
        if afs_balance == 0 and bs.get('Securities_HTM', 0) == 0 and 'Securities_Total' in bs:
            afs_balance = bs.get('Securities_Total', 0)

        # AOCI（税効果後）から税効果前含み損益を計算
        current_aoci = bs.get('AOCI', 0)
        tax_rate = self.params['TaxRate']

        # 税率が100%に近い場合のゼロ除算リスクを回避
        if abs(1 - tax_rate) > TOLERANCE:
            current_unrealized_gain_loss = current_aoci / (1 - tax_rate)
        else:
            current_unrealized_gain_loss = 0 # 税率100%ならAOCIは0になるはずだが念のため

        # 時価を計算（簿価 + 含み損益）
        market_value = afs_balance + current_unrealized_gain_loss
        market_value = max(0, market_value) # 時価はマイナスにならない

        # ヘアカットを適用
        hqla_securities = market_value * (1 - LCR_HQLA_HAIRCUT_AFS)
        total_hqla = hqla_level1 + hqla_securities

        # 純資金流出額
        net_cash_outflow = bs['Deposits_JA'] * LCR_NET_OUTFLOW_RATIO
        lcr = (total_hqla / net_cash_outflow) * 100 if net_cash_outflow > 0 else float('inf')

        return lcr, total_hqla

    def _calculate_metrics(self, bs, net_income, nii, rwa):
        securities_total = self._get_securities_balance(bs)
        total_assets = (bs.get('Deposits_Norinchukin_Mandatory', 0) +
                        bs.get('Deposits_Norinchukin_Voluntary', 0) +
                        bs.get('Loans', 0) + bs.get('OtherAssets', 0) + securities_total)

        earning_assets = total_assets - bs.get('OtherAssets', 0)

        # 会計上の自己資本（純資産）を簡易的に計算（規制資本 + AOCI）
        equity_accounting = bs.get('Equity_Regulatory', 0) + bs.get('AOCI', 0)

        # ROE（会計ベース）
        roe = (net_income / equity_accounting) * 100 if equity_accounting > 0 else 0
        nim = (nii / earning_assets) * 100 if earning_assets > 0 else 0

        # 自己資本比率（規制ベース）
        capital_adequacy_ratio = (bs.get('Equity_Regulatory', 0) / rwa) * 100 if rwa > 0 else 0

        # 流動性
        lcr, hqla = self._calculate_liquidity(bs)

        return {
            'ROE (%)': roe,
            'NIM (%)': nim,
            '自己資本比率 (対RWA) (%)': capital_adequacy_ratio,
            'LCR (%)': lcr,
            'HQLA (億円)': hqla / 1e8,
            '総資産': total_assets,
            'RWA': rwa,
        }

    def _calculate_base_case(self):
        bs = self.initial_bs.copy()
        pl = self.initial_pl.copy()
        params = self.params.copy()

        nii = self._calculate_nii(bs, params)
        ordinary_profit = nii + pl['OtherIncomeExpense'] - pl['Expenses'] - pl['CreditCost']

        profit_before_tax = ordinary_profit
        net_income = profit_before_tax * (1 - params['TaxRate'])
        rwa = self._calculate_rwa(bs)

        metrics = self._calculate_metrics(bs, net_income, nii, rwa)

        return {
            'NII': nii,
            'OrdinaryProfit': ordinary_profit,
            'NetIncome': net_income,
            'AOCI': bs['AOCI'],
            'BS': bs,
            **metrics
        }

    def run_simulation(self, scenario_params):
        bs = self.initial_bs.copy()
        pl = self.initial_pl.copy()
        rates = self.params.copy()

        p = scenario_params
        delta_rate = p['金利変化幅']
        target_duration = p['目標有価証券デュレーション']

        # 1. BSの変化
        initial_deposits_ja = self.initial_bs['Deposits_JA']
        bs['Deposits_JA'] = initial_deposits_ja * (1 + p['貯金流出率'])
        bs['Deposits_Norinchukin_Mandatory'] = bs['Deposits_JA'] * NORINCHUKIN_MANDATORY_RATIO

        deposit_change = bs['Deposits_JA'] - initial_deposits_ja
        allocatable_funds_change = deposit_change * (1 - NORINCHUKIN_MANDATORY_RATIO)

        initial_securities_balance = self._get_securities_balance(self.initial_bs)
        initial_allocatable_funds = self.initial_bs['Deposits_Norinchukin_Voluntary'] + initial_securities_balance + self.initial_bs['Loans']
        future_allocatable_funds = max(0, initial_allocatable_funds + allocatable_funds_change)

        bs['Loans'] = future_allocatable_funds * p['貸出金比率目標']
        securities_total = future_allocatable_funds * p['有価証券比率目標']
        # 任意預け金は残差で計算。浮動小数点誤差による微小なマイナスを防ぐため、max(0, ...) は維持
        bs['Deposits_Norinchukin_Voluntary'] = max(0, future_allocatable_funds - bs['Loans'] - securities_total)
        bs['Securities_HTM'] = securities_total * p['HTM比率目標']
        bs['Securities_AFS'] = securities_total - bs['Securities_HTM']
        bs.pop('Securities_Total', None) # Totalを削除し、HTM/AFSに分割

        # 2. 金利・コスト構造の変化
        rates['Rate_Deposits_Avg'] = max(0, self.params['Rate_Deposits_Avg'] + delta_rate * p['貯金金利連動率_平均'])
        rates['Rate_Norinchukin_Mandatory'] = max(0, self.params['Rate_Norinchukin_Mandatory'] + delta_rate * p['農中連動率_必須分'])
        rates['Rate_Norinchukin_Voluntary'] = max(0, self.params['Rate_Norinchukin_Voluntary'] + delta_rate * p['農中連動率_任意分'])

        market_impact_loan = delta_rate * p['貸出金利連動率_平均']
        strategy_impact_loan = p['貸出スプレッド変化']
        rates['Rate_Loans'] = max(0, self.params['Rate_Loans'] + market_impact_loan + strategy_impact_loan)

        strategy_impact_securities = p['有価証券リスクテイク（利回り向上幅）']
        rates['Rate_Securities'] = max(0, self.params['Rate_Securities'] + strategy_impact_securities)

        # 3. NIIの計算
        reinvestment_effect = 0
        # 再投資効果（年間償還額をデュレーションで簡易計算し、delta_rateで再投資される効果）
        if self.params['Securities_Duration'] > 0 and initial_securities_balance > 0:
            annual_reinvestment_amount = initial_securities_balance / self.params['Securities_Duration']
            reinvestment_effect = annual_reinvestment_amount * delta_rate

        # アセットスワップ効果（固定→変動化により、delta_rate分だけ利回りが変動する効果）
        asset_swap_ratio = p['アセットスワップ活用比率']
        asset_swap_amount = self._get_securities_balance(bs) * asset_swap_ratio
        asset_swap_effect = asset_swap_amount * delta_rate

        total_dynamic_effects = reinvestment_effect + asset_swap_effect
        future_nii = self._calculate_nii(bs, rates, total_dynamic_effects)

        # 4. その他PL
        if self.initial_bs['Loans'] > 0:
            credit_cost_ratio = self.initial_pl['CreditCost'] / self.initial_bs['Loans']
            pl['CreditCost'] = bs['Loans'] * credit_cost_ratio

        pl['OtherIncomeExpense'] = self.initial_pl['OtherIncomeExpense'] * (1 + p['役務収益等変化率'])
        pl['Expenses'] = self.initial_pl['Expenses'] * (1 + p['経費変化率'])

        # 5. 純利益と自己資本
        future_ordinary_profit = future_nii + pl['OtherIncomeExpense'] - pl['Expenses'] - pl['CreditCost']
        net_income = future_ordinary_profit * (1 - self.params['TaxRate'])

        # AOCI変動（デュレーションアプローチ、税効果考慮）
        aoci_change = - (bs['Securities_AFS'] * target_duration * delta_rate) * (1 - self.params['TaxRate'])
        future_aoci = self.initial_bs['AOCI'] + aoci_change
        bs['AOCI'] = future_aoci

        # 規制自己資本の変動（内部留保の変動分）
        retained_earnings_change = net_income - self.base_case_results['NetIncome']
        bs['Equity_Regulatory'] = self.initial_bs['Equity_Regulatory'] + retained_earnings_change

        # 6. RWA再計算
        # リスクテイクによるRW上昇
        rw_increase = strategy_impact_securities * RISK_TAKING_RWA_IMPACT_FACTOR
        rw_adjustment_factor = (self.params['RW_Securities'] + rw_increase) / self.params['RW_Securities'] if self.params['RW_Securities'] > 0 else 1.0
        future_rwa = self._calculate_rwa(bs, rw_adjustment_factor)

        # 7. 指標
        # 【修正】_calculate_metricsの引数を変更
        metrics = self._calculate_metrics(bs, net_income, future_nii, future_rwa)

        # 8. 要因分解 (NII)
        base_nii = self.base_case_results['NII']

        # ボリューム効果
        nii_after_volume = self._calculate_nii(bs, self.params)
        volume_effect = nii_after_volume - base_nii

        # 市場環境効果
        rates_market = self.params.copy()
        rates_market['Rate_Deposits_Avg'] = max(0, self.params['Rate_Deposits_Avg'] + delta_rate * p['貯金金利連動率_平均'])
        rates_market['Rate_Norinchukin_Mandatory'] = max(0, self.params['Rate_Norinchukin_Mandatory'] + delta_rate * p['農中連動率_必須分'])
        rates_market['Rate_Norinchukin_Voluntary'] = max(0, self.params['Rate_Norinchukin_Voluntary'] + delta_rate * p['農中連動率_任意分'])
        rates_market['Rate_Loans'] = max(0, self.params['Rate_Loans'] + market_impact_loan)
        # 有価証券利回りは市場効果を含まず、reinvestment_effectで調整

        nii_after_market = self._calculate_nii(bs, rates_market, reinvestment_effect)
        market_effect = nii_after_market - nii_after_volume

        # 戦略効果
        pricing_effect = bs['Loans'] * strategy_impact_loan
        risk_taking_effect = self._get_securities_balance(bs) * strategy_impact_securities
        alm_strategy_effect = asset_swap_effect

        return {
            '資金利益 (NII) (億円)': future_nii / 1e8,
            '当期純利益 (億円)': net_income / 1e8,
            'AOCI (億円)': future_aoci / 1e8,
            'AOCI変動 (億円)': aoci_change / 1e8,
            **{k: v for k, v in metrics.items() if k not in ['総資産', 'RWA']},
            '総資産 (億円)': metrics['総資産'] / 1e8,
            'RWA (億円)': metrics['RWA'] / 1e8,
            'BS': bs,
            'FutureAllocatableFunds': future_allocatable_funds,
            'NII_ベース (億円)': base_nii / 1e8,
            'NII_ボリューム効果 (億円)': volume_effect / 1e8,
            'NII_市場環境効果 (億円)': market_effect / 1e8,
            'NII_戦略効果_プライシング (億円)': pricing_effect / 1e8,
            'NII_戦略効果_リスクテイク (億円)': risk_taking_effect / 1e8,
            'NII_戦略効果_ALM (億円)': alm_strategy_effect / 1e8,
        }

# --- Streamlit アプリケーションのUI ---

def main():
    st.markdown("""
    <style>
        .stMetricValue { font-size: 24px; }
        .kpi-header {
            text-align: center; font-weight: bold; font-size: 18px;
            margin-bottom: 10px; padding: 5px; border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title('🌾 信連用 経営インパクトシミュレーター')
    st.caption('将来の環境変化（金利上昇・低下）と、それに対応する経営アクションの影響を分析します。')

    # --- サイドバー ---
    st.sidebar.header('1. データインプット')

    uploaded_file = st.sidebar.file_uploader("初期データ（CSV）", type="csv")
    sample_csv = get_sample_data_csv()

    if not uploaded_file:
        st.sidebar.download_button(
            label="サンプルCSV（標準モデル）",
            data=sample_csv.encode('utf-8-sig'),
            file_name="shinren_alm_input_sample.csv",
            mime="text/csv",
        )

    data_source = uploaded_file if uploaded_file else sample_csv
    if not uploaded_file:
        st.sidebar.info("サンプルデータ（標準モデル）を使用中")

    initial_data, error_message = load_data(data_source)
    if initial_data is None:
        st.error(error_message)
        st.stop()

    @st.cache_resource
    def get_simulator(data_hash):
        return ShinrenALMSimulator(initial_data)

    data_str = json.dumps(initial_data, sort_keys=True).encode()
    data_hash = hashlib.md5(data_str).hexdigest()
    simulator = get_simulator(data_hash)
    base_results = simulator.base_case_results

    # 初期比率計算
    initial_bs = simulator.initial_bs
    initial_securities = simulator._get_securities_balance(initial_bs)
    initial_allocatable_funds = initial_bs['Deposits_Norinchukin_Voluntary'] + initial_securities + initial_bs['Loans']

    if initial_allocatable_funds > 0:
        default_loan_ratio_pct = (initial_bs['Loans'] / initial_allocatable_funds) * 100
        default_securities_ratio_pct = (initial_securities / initial_allocatable_funds) * 100
    else:
        default_loan_ratio_pct = 0.0
        default_securities_ratio_pct = 0.0

    # --- シナリオ設定 ---
    st.sidebar.markdown("---")
    st.sidebar.header('2. 環境シナリオ（外部要因）')

    deposit_outflow_rate_input = st.sidebar.slider(
        '貯金流出率（年率）', min_value=-10.0, max_value=5.0, value=0.0, step=0.5, format='%.1f%%'
    )
    delta_rate_input = st.sidebar.slider(
        '市場金利変化幅 (ΔRate)', min_value=-1.0, max_value=2.0, value=0.0, step=0.05, format='%.2f%%'
    )

    expander_beta = st.sidebar.expander("市場金利への連動率（β値）")
    beta_deposit_avg_input = expander_beta.slider('貯金金利連動率', 0.0, 1.5, 1.0, 0.05)
    beta_nochu_mandatory_input = expander_beta.slider('農中預け金連動率（必須分）', 0.0, 1.5, 1.0, 0.05)
    beta_nochu_voluntary_input = expander_beta.slider('農中預け金連動率（任意分）', 0.0, 1.0, 1.0, 0.05)
    beta_loan_avg_input = expander_beta.slider('貸出金利連動率', 0.0, 1.0, 0.6, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.header('3. 経営アクション（内部要因）')

    st.sidebar.subheader("有価証券ポートフォリオ・ALM戦略")

    risk_taking_input = st.sidebar.slider(
        '1. リスクテイク: 利回り向上 (bps)', 0.0, 200.0, 0.0, 5.0,
        help="クレジットリスク等を取り、有価証券全体の平均利回りを向上させます。利回りが向上する一方、RWA（リスク量）も増加します。"
    )

    asset_swap_ratio_input = st.sidebar.slider(
        '2. ALM: アセットスワップ活用比率 (%)', 0.0, 100.0, 0.0, 5.0,
        help="有価証券の一部を変動金利化（固定金利受取→変動金利受取相当）し、金利変動リスクをヘッジします。金利上昇局面でNIIを改善させます。"
    )

    htm_ratio_input = st.sidebar.slider(
        '3. 構造: HTM比率目標 (%)', 0.0, 100.0, 0.0, 5.0,
        help="満期保有目的債券（HTM）の比率を高めると、金利変動による評価損益（AOCI）の変動を抑制（資本の安定化）できますが、流動性資産（HQLA）からは除外されるためLCRは低下します。"
    )
    target_duration_input = st.sidebar.slider(
        '4. 目標有価証券デュレーション（年）', 0.5, 10.0, float(simulator.params['Securities_Duration']), 0.5,
        help="ポートフォリオ全体の平均デュレーション（金利感応度）を調整します。短いほど金利変動リスク（AOCI変動）は小さくなります。"
    )

    st.sidebar.subheader("資産アロケーション戦略")
    # スライダーの操作性を考慮し、stepは0.1%とするが、初期値は高精度で設定される。表示も0.1%単位とする。
    SLIDER_FORMAT = '%.1f%%'
    
    loan_ratio_input = st.sidebar.slider(
        '5. 貸出金比率目標 (%)', 0.0, 100.0, default_loan_ratio_pct, 0.1, format=SLIDER_FORMAT,
        help="任意運用可能資金（必須預け金以外）のうち、貸出金に配分する比率を設定します。地域貢献や収益性向上に寄与しますが、与信リスクや流動性は低下します。"
    )
    # max_securities_ratioの計算は浮動小数点誤差の影響を受ける可能性があるため、内部では高精度を維持
    max_securities_ratio = 100.0 - loan_ratio_input
    
    # 初期値が最大値を超えないように調整（浮動小数点誤差対策）
    current_default_securities = default_securities_ratio_pct
    if current_default_securities > max_securities_ratio + TOLERANCE:
         current_default_securities = max_securities_ratio

    securities_ratio_input = st.sidebar.slider(
        '6. 有価証券比率目標 (%)', 0.0, max_securities_ratio, current_default_securities, 0.1, format=SLIDER_FORMAT,
        help="任意運用可能資金のうち、有価証券に配分する比率を設定します。収益性と流動性のバランスを調整します。"
    )
    # 農中（任意分）比率の計算と表示
    voluntary_nochu_ratio = max(0.0, 100.0 - loan_ratio_input - securities_ratio_input)
    st.sidebar.caption(f"→ 農林中金（任意分）比率: {voluntary_nochu_ratio:.1f}%")

    st.sidebar.subheader("コスト・収益多角化戦略")
    expenses_change_rate_input = st.sidebar.slider(
        '7. 経費変化率 (%)', -30.0, 30.0, 0.0, 1.0,
        help="業務効率化やコスト削減努力による経費（物件費・人件費）の変化率を設定します。"
        )
    other_income_change_rate_input = st.sidebar.slider(
        '8. 役務収益等変化率 (%)', -50.0, 100.0, 0.0, 5.0,
        help="投信販売手数料や為替手数料など、資金利益以外の収益（その他収支）の変化率を設定します。"
        )
    loan_spread_change_input = st.sidebar.slider(
        '9. 貸出プライシング: スプレッド変化 (bps)', -50.0, 100.0, 0.0, 5.0,
        help="市場金利の変動とは別に、貸出金利のスプレッド（上乗せ金利）を変更します。競争環境やリスク選好度に応じて調整します。"
        )

    # パラメータ集約
    current_params = {
        '金利変化幅': delta_rate_input / 100,
        '貯金流出率': deposit_outflow_rate_input / 100,
        '貯金金利連動率_平均': beta_deposit_avg_input,
        '農中連動率_必須分': beta_nochu_mandatory_input,
        '農中連動率_任意分': beta_nochu_voluntary_input,
        '貸出金利連動率_平均': beta_loan_avg_input,
        '有価証券リスクテイク（利回り向上幅）': risk_taking_input / 10000,
        'アセットスワップ活用比率': asset_swap_ratio_input / 100,
        'HTM比率目標': htm_ratio_input / 100,
        '目標有価証券デュレーション': target_duration_input,
        '貸出金比率目標': loan_ratio_input / 100,
        '有価証券比率目標': securities_ratio_input / 100,
        '貸出スプレッド変化': loan_spread_change_input / 10000,
        '経費変化率': expenses_change_rate_input / 100,
        '役務収益等変化率': other_income_change_rate_input / 100,
    }

    scenario_results = simulator.run_simulation(current_params)

    # --- メイン画面の描画 ---
    render_kpi_summary(base_results, scenario_results)
    st.markdown("---")

    st.header("2. 要因分析とバランスシート変化")
    col_analysis_left, col_analysis_right = st.columns([1.2, 1])

    with col_analysis_left:
        render_nii_wf(scenario_results)

    with col_analysis_right:
        render_bs_change(base_results, scenario_results)

    st.markdown("---")

    st.header("3. 戦略的意思決定の示唆")
    tab1, tab2 = st.tabs(["3.1 HTM戦略のトレードオフ（資本安定性 vs 流動性）", "3.2 リスクテイク戦略のトレードオフ（収益性 vs 健全性）"])
    with tab1:
        render_htm_tradeoff_analysis(simulator, current_params, delta_rate_input)
    with tab2:
        render_risk_taking_analysis(simulator, current_params)

    st.markdown("---")
    st.header("4. 結果のダウンロードと詳細情報")

    # ダウンロードデータの準備（BOM付きUTF-8対応）
    csv_bytes = prepare_download_data(base_results, scenario_results, current_params)
    st.download_button(
        label="CSVでダウンロード",
        data=csv_bytes,
        file_name="shinren_alm_simulation_results.csv",
        mime="text/csv",
    )

    # 詳細情報の表示
    with st.expander("モデル解説・初期データ確認", expanded=False):
        render_info(simulator)

# --- 描画関数群 ---

def render_kpi_summary(base_results, scenario_results):
    st.header("1. シミュレーション結果サマリー")
    col1, col2, col3, col4 = st.columns(4)

    def format_delta(delta, unit, precision=1):
        # 浮動小数点誤差による微小な値を0として扱う
        if abs(delta) < TOLERANCE:
            delta = 0.0
        fmt = f"{{:+.{precision}f}} {unit}"
        return fmt.format(delta)

    with col1:
        st.markdown(f'<div class="kpi-header" style="background-color: {COLOR_TARGET}20;">目標達成度</div>', unsafe_allow_html=True)
        net_income = scenario_results['当期純利益 (億円)']
        delta_ni = net_income - base_results['NetIncome']/1e8
        st.metric("当期純利益 (億円)", f"{net_income:.1f}", format_delta(delta_ni, "億円"))

        roe = scenario_results['ROE (%)']
        delta_roe = roe - base_results['ROE (%)']
        st.metric("ROE (%) (会計ベース)", f"{roe:.2f}", format_delta(delta_roe, "pt", precision=2))

    with col2:
        st.markdown(f'<div class="kpi-header" style="background-color: {COLOR_PROFIT}20;">収益性</div>', unsafe_allow_html=True)
        nii = scenario_results['資金利益 (NII) (億円)']
        delta_nii = nii - base_results['NII']/1e8
        st.metric("資金利益 (NII) (億円)", f"{nii:.1f}", format_delta(delta_nii, "億円"))

        nim = scenario_results['NIM (%)']
        delta_nim = nim - base_results['NIM (%)']
        st.metric("NIM (%)", f"{nim:.2f}", format_delta(delta_nim, "pt", precision=2))

    with col3:
        st.markdown(f'<div class="kpi-header" style="background-color: {COLOR_CAPITAL}20;">健全性</div>', unsafe_allow_html=True)
        car = scenario_results['自己資本比率 (対RWA) (%)']
        delta_car = car - base_results['自己資本比率 (対RWA) (%)']
        st.metric("自己資本比率 (対RWA) (%)", f"{car:.2f}", format_delta(delta_car, "pt", precision=2))

        aoci = scenario_results['AOCI (億円)']
        delta_aoci = aoci - base_results['AOCI']/1e8
        st.metric("評価損益 (AOCI) (億円)", f"{aoci:.1f}", format_delta(delta_aoci, "億円"), delta_color="inverse")

    with col4:
        st.markdown(f'<div class="kpi-header" style="background-color: {COLOR_LIQUIDITY}20;">流動性</div>', unsafe_allow_html=True)
        lcr = scenario_results['LCR (%)']
        delta_lcr = lcr - base_results['LCR (%)']
        st.metric("LCR (%) (簡易試算)", f"{lcr:.1f}", format_delta(delta_lcr, "pt"))

        hqla = scenario_results['HQLA (億円)']
        delta_hqla = hqla - base_results['HQLA (億円)']
        # HQLAは整数で表示
        st.metric("HQLA残高 (億円)", f"{hqla:,.0f}", format_delta(delta_hqla, "億円", precision=0))

def render_nii_wf(results):
    st.subheader("資金利益（NII）変動の要因分析")

    labels = [
        "現状NII", "ボリューム効果", "市場金利変動", "リスクテイク戦略",
        "スワップ戦略", "プライシング戦略", "想定NII"
    ]

    values = [
        results['NII_ベース (億円)'], results['NII_ボリューム効果 (億円)'],
        results['NII_市場環境効果 (億円)'], results['NII_戦略効果_リスクテイク (億円)'],
        results['NII_戦略効果_ALM (億円)'], results['NII_戦略効果_プライシング (億円)'],
        results['資金利益 (NII) (億円)']
    ]

    # 微小な値を0に丸める
    values = [0.0 if abs(v) < TOLERANCE else v for v in values]

    measures = ["absolute"] + ["relative"] * 5 + ["total"]
    text_data = [f"{v:+.1f}" if measure == 'relative' else f"{v:.1f}" for v, measure in zip(values, measures)]

    fig = go.Figure(go.Waterfall(
        name = "NII変動要因", orientation = "v", measure = measures, x = labels,
        textposition = "outside", text = text_data, y = values,
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        increasing = {"marker":{"color":COLOR_INCREASE}},
        decreasing = {"marker":{"color":COLOR_DECREASE}},
        totals = {"marker":{"color":COLOR_JA_GREEN}},
    ))

    fig.update_layout(
        title="資金利益（NII）ウォーターフォール（単位：億円）", showlegend=False, height=500,
        margin=dict(l=20, r=20, t=60, b=80), yaxis_title="金額 (億円)", xaxis=dict(tickangle=-45)
    )
    st.plotly_chart(fig, use_container_width=True)

def render_bs_change(base_results, scenario_results):
    st.subheader("バランスシート（資産サイド）の変化")
    base_bs = base_results['BS']
    scenario_bs = scenario_results['BS']

    def prepare_bs_data(bs_data, is_base=False):
        # 表示用に億円単位で返す（グラフ表示時にPlotlyが丸める）
        return {
            '必須預け金(農中)': bs_data.get('Deposits_Norinchukin_Mandatory', 0) / 1e8,
            '任意預け金(農中)': bs_data.get('Deposits_Norinchukin_Voluntary', 0) / 1e8,
            '有価証券(HTM)': (0 if is_base else bs_data.get('Securities_HTM', 0)) / 1e8,
            '有価証券(AFS/その他)': (bs_data.get('Securities_Total', 0) if is_base else bs_data.get('Securities_AFS', 0)) / 1e8,
            '貸出金': bs_data.get('Loans', 0) / 1e8,
            'その他資産': bs_data.get('OtherAssets', 0) / 1e8
        }

    base_data = prepare_bs_data(base_bs, is_base=True)
    scenario_data = prepare_bs_data(scenario_bs)
    # スタックバーの積み上げ順序（下から上へ）
    asset_categories = ['その他資産', '貸出金', '有価証券(HTM)', '有価証券(AFS/その他)', '任意預け金(農中)', '必須預け金(農中)']

    fig = go.Figure()
    for cat in asset_categories:
        base_val = base_data.get(cat, 0)
        scen_val = scenario_data.get(cat, 0)
        text_fmt = "{:,.0f}" if (base_val > 500 or scen_val > 500) else ""

        fig.add_trace(go.Bar(
            name=cat, x=['現状', 'シナリオ後'], y=[base_val, scen_val],
            marker_color=BS_COLORS.get(cat, '#A9A9A9'),
            text=[text_fmt.format(v) for v in [base_val, scen_val]],
            textposition='inside', insidetextanchor='middle'
        ))

    # 総資産額の表示
    total_assets_base = sum(base_data.values())
    total_assets_scenario = sum(scenario_data.values())

    fig.add_annotation(x='現状', y=total_assets_base, text=f"<b>総資産: {total_assets_base:,.0f}</b>", showarrow=False, yshift=15)
    fig.add_annotation(x='シナリオ後', y=total_assets_scenario, text=f"<b>総資産: {total_assets_scenario:,.0f}</b>", showarrow=False, yshift=15)

    # legend_traceorder="reversed" を指定し、凡例の順序をスタックバーの順序と一致させる
    fig.update_layout(barmode='stack', title="資産構成の変化（単位：億円）", yaxis_title="金額 (億円)", height=500, legend_traceorder="reversed")
    st.plotly_chart(fig, use_container_width=True)

def render_htm_tradeoff_analysis(simulator, current_params, delta_rate):
    htm_steps = np.linspace(0, 100, 11)
    results_list = []
    params = current_params.copy()

    for htm_ratio in htm_steps:
        params['HTM比率目標'] = htm_ratio / 100
        res = simulator.run_simulation(params)
        results_list.append({
            'HTM比率 (%)': htm_ratio,
            'LCR (%)': res['LCR (%)'],
            'AOCI (億円)': res['AOCI (億円)'],
        })

    df = pd.DataFrame(results_list)
    fig = go.Figure(go.Scatter(
        x=df['AOCI (億円)'], y=df['LCR (%)'], mode='lines+markers+text',
        text=df['HTM比率 (%)'].apply(lambda x: f'{x:.0f}%'), textposition="top center",
        marker=dict(size=10, color=df['HTM比率 (%)'], colorscale='Viridis', showscale=True, colorbar_title="HTM比率"),
        line=dict(dash='dot', color='gray')
    ))

    # グラフタイトルの動的生成
    title_text = f"HTM比率のトレードオフ（金利{delta_rate:+.2f}%変動時）"

    fig.update_layout(title=title_text, xaxis_title="資本安定性：AOCI (億円)", yaxis_title="流動性：LCR (%)", height=500)
    st.plotly_chart(fig, use_container_width=True)

def render_risk_taking_analysis(simulator, current_params):
    steps = np.linspace(0, 200, 11)
    results_list = []
    params = current_params.copy()

    for bps in steps:
        params['有価証券リスクテイク（利回り向上幅）'] = bps / 10000
        res = simulator.run_simulation(params)
        results_list.append({
            '利回り向上幅 (bps)': bps,
            'NII (億円)': res['資金利益 (NII) (億円)'],
            '自己資本比率 (%)': res['自己資本比率 (対RWA) (%)'],
        })

    df = pd.DataFrame(results_list)
    fig = go.Figure(go.Scatter(
        x=df['自己資本比率 (%)'], y=df['NII (億円)'], mode='lines+markers+text',
        text=df['利回り向上幅 (bps)'].apply(lambda x: f'{x:.0f}bps'), textposition="top center",
        marker=dict(size=10, color=df['利回り向上幅 (bps)'], colorscale='Plasma', showscale=True, colorbar_title="利回り向上幅(bps)"),
        line=dict(dash='dot', color='gray')
    ))
    fig.update_layout(title="リスクテイク戦略のトレードオフ（収益性 vs 健全性）", xaxis_title="健全性：自己資本比率 (%)", yaxis_title="収益性：NII (億円)", height=500)
    st.plotly_chart(fig, use_container_width=True)

def prepare_download_data(base, scenario, params):
    """CSVデータを生成し、UTF-8-SIGでエンコードしたバイト列を返す"""
    summary_data = {
        '項目': ['当期純利益 (億円)', 'ROE (%) (会計ベース)', '資金利益 (NII) (億円)', 'NIM (%)', '自己資本比率 (対RWA) (%)', '評価損益 (AOCI) (億円)', 'LCR (%)', 'HQLA残高 (億円)', '総資産 (億円)', 'RWA (億円)'],
        '現状': [base['NetIncome']/1e8, base['ROE (%)'], base['NII']/1e8, base['NIM (%)'], base['自己資本比率 (対RWA) (%)'], base['AOCI']/1e8, base['LCR (%)'], base['HQLA (億円)'], base['総資産']/1e8, base['RWA']/1e8],
        'シナリオ後': [scenario['当期純利益 (億円)'], scenario['ROE (%)'], scenario['資金利益 (NII) (億円)'], scenario['NIM (%)'], scenario['自己資本比率 (対RWA) (%)'], scenario['AOCI (億円)'], scenario['LCR (%)'], scenario['HQLA (億円)'], scenario['総資産 (億円)'], scenario['RWA (億円)']]
    }

    # メモリ上のテキストバッファに書き込み
    output = io.StringIO()
    output.write("シミュレーション結果サマリー\n")
    # 浮動小数点誤差を丸めて出力
    pd.DataFrame(summary_data).round(4).to_csv(output, index=False)
    output.write("\nパラメータ設定値\n")
    pd.DataFrame({'パラメータ': params.keys(), '設定値': params.values()}).to_csv(output, index=False)

    # 文字列を取得し、utf-8-sig (BOM付き) でエンコードしてバイト列にする
    csv_str = output.getvalue()
    return csv_str.encode('utf-8-sig')

def render_info(simulator):
    """モデル解説と初期データを詳細に表示する"""
    st.subheader("モデルの前提と解説")
    p = simulator.params
    st.markdown(f"""
    本シミュレーターは、以下の前提に基づき計算を行っています。

    - **収益構造**: NII（資金利益）中心の構造。役務収益等の変化もシミュレーション可能。
    - **自己資本比率（国内基準前提）**: RWA対比で計算。国内基準に基づき、AOCI変動は規制自己資本比率の分子（Equity_Regulatory）に影響しません。当期純利益の変動（内部留保）のみ影響します。
    - **ROE（会計ベース）**: 会計上の自己資本（純資産 = 規制資本分子 + AOCI）ベースで計算しています。
    - **リスクテイク戦略**: 有価証券の利回り向上（クレジットリスク取得等）は、RWAの増加（RW上昇）を伴います。影響度は簡易的に線形でモデル化しています（利回り1%向上あたりRWが{RISK_TAKING_RWA_IMPACT_FACTOR}%pt上昇）。
    - **ALM戦略（アセットスワップ）**: 有価証券の一部を変動金利化（固定金利→変動金利）する効果をモデル化。金利上昇局面でNIIを改善させます。
    - **金利変動と再投資**: 市場金利の変化に対する各種金利の連動率（β値）を設定。金利は0%下限。有価証券の再投資効果は、デュレーションに基づき簡易的に計算しています。
    - **必須預け金**: JAからの貯金の{NORINCHUKIN_MANDATORY_RATIO*100:.0f}%を農中への必須預け金と仮定。
    - **流動性（LCR/HQLA）**: 任意預け金とAFS（時価評価後・ヘアカット{LCR_HQLA_HAIRCUT_AFS*100:.0f}%）をHQLAとみなし簡易計算。HTMはHQLAに含まれません。時価はAOCIから逆算して整合性を確保しています。
    """)

    st.subheader("初期データ確認（ベースケース）")

    # 初期BS表示
    st.markdown("**貸借対照表（資産サイド）とRWA構成**")
    bs = simulator.initial_bs

    bs_data = {
        '項目': ['必須預け金', '任意預け金', '有価証券', '貸出金', 'その他資産', '合計'],
        '金額（億円）': [
            bs['Deposits_Norinchukin_Mandatory']/1e8, bs['Deposits_Norinchukin_Voluntary']/1e8,
            simulator._get_securities_balance(bs)/1e8, bs['Loans']/1e8, bs['OtherAssets']/1e8,
            simulator.base_case_results['総資産']/1e8
        ],
        'RW（推定）': [
            f"{p['RW_Deposits']*100:.1f}%", f"{p['RW_Deposits']*100:.1f}%",
            f"{p['RW_Securities']*100:.1f}%", f"{p['RW_Loans']*100:.1f}%", f"{p['RW_OtherAssets']*100:.2f}%", '-'
        ]
    }
    # 表示フォーマットを適用
    st.table(pd.DataFrame(bs_data).set_index('項目').style.format({'金額（億円）': '{:,.0f}'}))
    st.info(f"RWA合計: {simulator.base_case_results['RWA']/1e8:,.0f}億円 (うち固定RWA: {p['FixedRWA']/1e8:,.0f}億円)")

    # 初期PL表示
    st.markdown("**損益計算書（構造）と主要利回り**")
    pl = simulator.initial_pl
    pl_data = {
        '項目': ['資金利益 (NII)', '+) その他収支', '-) 経費', '-) 与信コスト', '=) 経常利益', '当期純利益'],
        '金額（億円）': [
            pl['NII']/1e8, pl['OtherIncomeExpense']/1e8, pl['Expenses']/1e8, pl['CreditCost']/1e8,
            pl['OrdinaryProfit']/1e8, pl['NetIncome']/1e8
        ]
    }

    param_data = {
        '項目': ['必須預け金利回り', '任意預け金利回り', '有価証券利回り', '貸出金利回り', '貯金利回り（平均）'],
        '値 (%)': [
            f"{p['Rate_Norinchukin_Mandatory']*100:.3f}", f"{p['Rate_Norinchukin_Voluntary']*100:.3f}",
            f"{p['Rate_Securities']*100:.2f}", f"{p['Rate_Loans']*100:.2f}", f"{p['Rate_Deposits_Avg']*100:.2f}"
        ]
    }

    col1, col2 = st.columns(2)
    with col1:
        st.table(pd.DataFrame(pl_data).set_index('項目').style.format({'金額（億円）': '{:,.1f}'}))
    with col2:
        st.table(pd.DataFrame(param_data).set_index('項目'))

if __name__ == '__main__':
    np.seterr(all='ignore')
    main()