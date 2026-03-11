import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.patches import Patch
from IPython.display import display

# --- Matplotlibの日本語フォント設定---
def set_japanese_font():
    jp_fonts = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Noto Sans CJK JP', 'MS Gothic', 'IPAexGothic']
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    font_set = False
    for font in jp_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            font_set = True
            break
    if not font_set:
        pass

set_japanese_font()
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.axisbelow'] = True  # グリッド線をグラフの背面に設定
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

# =========================================================
# 1. 各種マスタとインプットデータの生成（4月開始・12ヶ月分）
# =========================================================

# freq='M' から freq='ME' へ修正
months = pd.date_range(start='2025-03-31', end='2026-03-31', freq='ME').strftime('%Y-%m')
np.random.seed(42)

# ① マクロ指標データ（政策金利初期パスを0.75%に修正）
macro_data = []
current_rates = {'政策金利': 0.0075, '国内金利': 0.008, '米国金利': 0.040}
current_prices = {'国内株価': 2500.0, '米国株価': 5000.0, 'JREIT': 1800.0, '投信価格': 10000.0, 'USDJPY': 145.0}

for m in months:
    macro_data.append({
        '年月': m,
        '政策金利': current_rates['政策金利'],
        '国内金利': current_rates['国内金利'],
        '米国金利': current_rates['米国金利'],
        '国内株価': current_prices['国内株価'],
        '米国株価': current_prices['米国株価'],
        'JREIT': current_prices['JREIT'],
        '投信価格': current_prices['投信価格'],
        'USDJPY': current_prices['USDJPY']
    })
    current_rates['国内金利'] += np.random.normal(0, 0.0003)
    current_rates['米国金利'] += np.random.normal(0, 0.0010)
    current_prices['国内株価'] *= (1 + np.random.normal(0.005, 0.03))
    current_prices['米国株価'] *= (1 + np.random.normal(0.008, 0.04))
    current_prices['JREIT'] *= (1 + np.random.normal(0.003, 0.025))
    current_prices['投信価格'] *= (1 + np.random.normal(0.006, 0.02))
    current_prices['USDJPY'] *= (1 + np.random.normal(0.000, 0.015))

df_macro = pd.DataFrame(macro_data)

# ② 銘柄マスタ（8資産）
df_master = pd.DataFrame([
    {'銘柄コード': 'B001', '銘柄名': '国内国債10年',   '資産区分': '内債_固定'},
    {'銘柄コード': 'B002', '銘柄名': '国内変動付社債', '資産区分': '内債_変動'},
    {'銘柄コード': 'B003', '銘柄名': '米国債H有',       '資産区分': '外債H有_固定'},
    {'銘柄コード': 'B004', '銘柄名': '米国変動債H有',   '資産区分': '外債H有_変動'},
    {'銘柄コード': 'B005', '銘柄名': '米国債H無',       '資産区分': '外債Hなし'},
    {'銘柄コード': 'E001', '銘柄名': 'TOPIX連動ETF',    '資産区分': '内株'},
    {'銘柄コード': 'R001', '銘柄名': 'J-REITファンド',  '資産区分': 'J-REIT'},
    {'銘柄コード': 'F001', '銘柄名': 'バランス型投信',  '資産区分': '投信'}
])

# ③ ベンチマーク設定マスタ
df_bm_attr = pd.DataFrame([
    {'資産区分': '内債_固定',     'BMウェイト': 0.30, 'デュレーション': 8.0, 'ベータ': 0.0, '参照金利': '国内金利', '参照株価': None,     '参照為替': None},
    {'資産区分': '内債_変動',     'BMウェイト': 0.10, 'デュレーション': 0.5, 'ベータ': 0.0, '参照金利': '国内金利', '参照株価': None,     '参照為替': None},
    {'資産区分': '外債H有_固定',  'BMウェイト': 0.10, 'デュレーション': 7.0, 'ベータ': 0.0, '参照金利': '米国金利', '参照株価': None,     '参照為替': None},
    {'資産区分': '外債H有_変動',  'BMウェイト': 0.05, 'デュレーション': 0.5, 'ベータ': 0.0, '参照金利': '米国金利', '参照株価': None,     '参照為替': None},
    {'資産区分': '外債Hなし',     'BMウェイト': 0.05, 'デュレーション': 7.0, 'ベータ': 0.0, '参照金利': '米国金利', '参照株価': None,     '参照為替': 'USDJPY'},
    {'資産区分': '内株',         'BMウェイト': 0.15, 'デュレーション': 0.0, 'ベータ': 1.0, '参照金利': None,     '参照株価': '国内株価', '参照為替': None},
    {'資産区分': 'J-REIT',       'BMウェイト': 0.05, 'デュレーション': 0.0, 'ベータ': 1.0, '参照金利': None,     '参照株価': 'JREIT',    '参照為替': None},
    {'資産区分': '投信',         'BMウェイト': 0.20, 'デュレーション': 0.0, 'ベータ': 1.0, '参照金利': None,     '参照株価': '投信価格', '参照為替': None},
])

# ④ ベンチマーク・リターンの生成
bm_data_list = []
for i in range(1, len(months)):
    prev_m = months[i-1]
    curr_m = months[i]
    mac_prev = df_macro[df_macro['年月'] == prev_m].iloc[0]
    mac_curr = df_macro[df_macro['年月'] == curr_m].iloc[0]
    
    for _, row in df_bm_attr.iterrows():
        ret = 0.0
        if row['参照金利']:
            rate_diff = mac_curr[row['参照金利']] - mac_prev[row['参照金利']]
            ret += -row['デュレーション'] * rate_diff + (mac_prev[row['参照金利']] / 12)
        if row['参照株価']:
            price_ret = (mac_curr[row['参照株価']] / mac_prev[row['参照株価']]) - 1
            ret += row['ベータ'] * price_ret
        if row['参照為替']:
            fx_ret = (mac_curr[row['参照為替']] / mac_prev[row['参照為替']]) - 1
            ret += fx_ret
            
        bm_data_list.append({'年月': curr_m, '資産区分': row['資産区分'], 'BMリターン': ret})

df_bm = pd.DataFrame(bm_data_list)
df_bm = pd.merge(df_bm, df_bm_attr[['資産区分', 'BMウェイト']], on='資産区分', how='left')

# ⑤ 月末残高と期中取引の生成
holdings_state = {
    'B001': {'qty': 30000, 'price': 100.0}, 'B002': {'qty': 10000, 'price': 100.0},
    'B003': {'qty': 10000, 'price': 110.0}, 'B004': {'qty': 5000,  'price': 110.0},
    'B005': {'qty': 5000,  'price': 110.0}, 'E001': {'qty': 1000,  'price': 1500.0},
    'R001': {'qty': 250,   'price': 2000.0}, 'F001': {'qty': 2000,  'price': 10000.0}
}

holdings_data = []
transactions_data = []

for code, state in holdings_state.items():
    holdings_data.append({'年月': months[0], '銘柄コード': code, '月末数量': state['qty'], '時価単価': state['price']})

for m in months[1:]:
    bm_month = df_bm[df_bm['年月'] == m].set_index('資産区分')
    
    for code, state in holdings_state.items():
        asset = df_master[df_master['銘柄コード'] == code]['資産区分'].iloc[0]
        bm_ret = bm_month.loc[asset, 'BMリターン']
        indiv_ret = bm_ret + np.random.normal(0, 0.001) 
        state['price'] *= (1 + indiv_ret)
        
        trade_qty = 0
        is_buy = True
        
        if indiv_ret < -0.015: 
            trade_qty = int(state['qty'] * 0.05)
            is_buy = True
        elif indiv_ret > 0.02: 
            trade_qty = int(state['qty'] * 0.03)
            is_buy = False
        elif np.random.rand() > 0.8:
            trade_qty = int(state['qty'] * 0.01)
            is_buy = np.random.rand() > 0.4
            
        if trade_qty > 0:
            trade_type = '買' if is_buy else '売'
            cost_rate = 0.001 if code in ['E001', 'R001', 'F001'] else 0.0001
            cost = trade_qty * state['price'] * cost_rate
            transactions_data.append({'年月': m, '銘柄コード': code, '区分': trade_type, '数量': trade_qty, '単価': state['price'], '金額': trade_qty * state['price'], 'コスト': cost})
            state['qty'] += trade_qty if is_buy else -trade_qty

        if np.random.rand() > 0.7:
            transactions_data.append({'年月': m, '銘柄コード': code, '区分': 'インカム', '数量': 0, '単価': 0, '金額': state['qty'] * state['price'] * 0.002, 'コスト': 0})

        holdings_data.append({'年月': m, '銘柄コード': code, '月末数量': state['qty'], '時価単価': state['price']})

df_holdings = pd.DataFrame(holdings_data)
df_transactions = pd.DataFrame(transactions_data)


# =========================================================
# 2. 計算エンジン（単月の要因分解を行う関数）
# =========================================================
def calculate_attribution(target_month, prev_month, df_h, df_t, df_b, df_mac, df_mast, df_attr):
    policy_rate = df_mac[df_mac['年月'] == target_month]['政策金利'].iloc[0]
    A = (policy_rate + 0.008) / 12  
    
    h_start = df_h[df_h['年月'] == prev_month].copy().rename(columns={'月末数量': '期首数量', '時価単価': '期首単価'})
    h_start = pd.merge(h_start, df_mast[['銘柄コード', '資産区分']], on='銘柄コード', how='left')
    h_start['期首評価額'] = h_start['期首数量'] * h_start['期首単価']
    h_end = df_h[df_h['年月'] == target_month].copy().rename(columns={'月末数量': '期末数量', '時価単価': '期末単価'})
    
    t_month = df_t[df_t['年月'] == target_month].copy()
    t_month = pd.merge(t_month, df_mast[['銘柄コード', '資産区分']], on='銘柄コード', how='left')
    
    income = t_month[t_month['区分'] == 'インカム'].groupby('銘柄コード')['金額'].sum().reset_index().rename(columns={'金額':'インカム'})
    trades = t_month[t_month['区分'].isin(['買', '売'])].copy()
    trades['数量増減'] = trades.apply(lambda x: x['数量'] if x['区分'] == '買' else -x['数量'], axis=1)
    trades['受渡金額'] = trades.apply(lambda x: -x['金額'] if x['区分'] == '買' else x['金額'], axis=1)
    trade_summary = trades.groupby('銘柄コード').agg({'数量増減': 'sum', '受渡金額': 'sum', 'コスト': 'sum'}).reset_index()

    df_eval = pd.merge(h_start, h_end[['銘柄コード', '期末単価']], on='銘柄コード', how='left')
    df_eval['BH期末評価額'] = df_eval['期首数量'] * df_eval['期末単価']
    df_eval = pd.merge(df_eval, income, on='銘柄コード', how='left').fillna(0)
    df_eval['BH損益額'] = (df_eval['BH期末評価額'] - df_eval['期首評価額']) + df_eval['インカム']
    
    df_eval = pd.merge(df_eval, trade_summary, on='銘柄コード', how='left').fillna(0)
    df_eval['実際期末評価額'] = (df_eval['期首数量'] + df_eval['数量増減']) * df_eval['期末単価']
    df_eval['実際損益額'] = (df_eval['実際期末評価額'] - df_eval['期首評価額']) + df_eval['受渡金額'] + df_eval['インカム'] - df_eval['コスト']
    df_eval['タイミング効果額'] = df_eval['実際損益額'] - df_eval['BH損益額'] + df_eval['コスト']

    port_start_value = df_eval['期首評価額'].sum()
    B = df_eval['実際損益額'].sum() / port_start_value
    T = df_eval['タイミング効果額'].sum() / port_start_value
    Cost = -df_eval['コスト'].sum() / port_start_value
    
    asset_eval = df_eval.groupby('資産区分').agg({'期首評価額': 'sum', 'BH損益額': 'sum'}).reset_index()
    asset_eval['期首ウェイト'] = asset_eval['期首評価額'] / port_start_value
    asset_eval['BHリターン'] = asset_eval['BH損益額'] / asset_eval['期首評価額']
    
    asset_tc = df_eval.groupby('資産区分').agg({'タイミング効果額': 'sum', 'コスト': 'sum', '実際損益額': 'sum'}).reset_index()
    asset_tc['タイミング効果額'] /= port_start_value
    asset_tc['コスト'] = -asset_tc['コスト'] / port_start_value
    asset_tc['実績B'] = asset_tc['実際損益額'] / port_start_value
    asset_eval = pd.merge(asset_eval, asset_tc, on='資産区分', how='left').fillna(0)
    
    bm_month = df_b[df_b['年月'] == target_month].copy()
    asset_eval = pd.merge(asset_eval, bm_month, on='資産区分', how='inner')
    
    C = (asset_eval['BMウェイト'] * asset_eval['BMリターン']).sum()
    
    asset_eval['アロケーション効果'] = (asset_eval['期首ウェイト'] - asset_eval['BMウェイト']) * (asset_eval['BMリターン'] - C)
    asset_eval['銘柄選択効果'] = asset_eval['BMウェイト'] * (asset_eval['BHリターン'] - asset_eval['BMリターン'])
    
    asset_eval['ベンチマークC'] = asset_eval['BMウェイト'] * asset_eval['BMリターン']
    asset_eval['資産別B_C'] = asset_eval['実績B'] - asset_eval['ベンチマークC']
    asset_eval['資産別行動_誤差'] = asset_eval['資産別B_C'] - (asset_eval['アロケーション効果'] + asset_eval['銘柄選択効果'] + asset_eval['タイミング効果額'] + asset_eval['コスト'])
    
    alloc_eff = asset_eval['アロケーション効果'].sum()
    select_eff = asset_eval['銘柄選択効果'].sum()
    action_unexplained = asset_eval['資産別行動_誤差'].sum()
    
    mac_prev = df_mac[df_mac['年月'] == prev_month].iloc[0]
    mac_curr = df_mac[df_mac['年月'] == target_month].iloc[0]
    env_interest, env_stock = 0.0, 0.0
    
    for _, row in df_attr.iterrows():
        bm_weight = row['BMウェイト']
        if row['参照金利']:
            rate_diff = mac_curr[row['参照金利']] - mac_prev[row['参照金利']]
            factor_ret = -row['デュレーション'] * rate_diff + (mac_prev[row['参照金利']] / 12)
            env_interest += bm_weight * factor_ret
        if row['参照株価']:
            price_ret = (mac_curr[row['参照株価']] / mac_prev[row['参照株価']]) - 1
            factor_ret = row['ベータ'] * price_ret
            env_stock += bm_weight * factor_ret
        if row['参照為替']:
            fx_ret = (mac_curr[row['参照為替']] / mac_prev[row['参照為替']]) - 1
            env_stock += bm_weight * fx_ret 
            
    env_other = C - (env_interest + env_stock)

    res_dict = {
        '年月': target_month,
        '目標(A)': A, '実績(B)': B, 'ベンチマーク(C)': C,
        'マクロ環境(C-A)': C - A, '行動(B-C)': B - C,
        '環境_金利': env_interest, '環境_株式等': env_stock, '環境_目標控除': -A, '環境_誤差': env_other,
        '行動_配分': alloc_eff, '行動_銘柄': select_eff, '行動_売買': T, '行動_コスト': Cost, '行動_誤差': action_unexplained
    }
    
    for _, row in df_attr.iterrows():
        ast = row['資産区分']
        if ast in asset_eval['資産区分'].values:
            ast_row = asset_eval[asset_eval['資産区分'] == ast].iloc[0]
            res_dict[f'配分_{ast}'] = ast_row['アロケーション効果']
            res_dict[f'銘柄_{ast}'] = ast_row['銘柄選択効果']
            res_dict[f'売買_{ast}'] = ast_row['タイミング効果額']
            res_dict[f'コスト_{ast}'] = ast_row['コスト']
            res_dict[f'誤差_{ast}'] = ast_row['資産別行動_誤差']
        else:
            res_dict[f'配分_{ast}'] = res_dict[f'銘柄_{ast}'] = res_dict[f'売買_{ast}'] = res_dict[f'コスト_{ast}'] = res_dict[f'誤差_{ast}'] = 0.0

    return res_dict

results = []
for i in range(1, len(months)):
    res = calculate_attribution(months[i], months[i-1], df_holdings, df_transactions, df_bm, df_macro, df_master, df_bm_attr)
    results.append(res)
df_results = pd.DataFrame(results)

# =========================================================
# 3. カリーノ法（Carino Smoothing）による累積計算
# =========================================================
def calculate_cumulative_carino(df_res, attr_list):
    A_cum = np.prod(1 + df_res['目標(A)']) - 1
    C_cum = np.prod(1 + df_res['ベンチマーク(C)']) - 1
    B_cum = np.prod(1 + df_res['実績(B)']) - 1
    
    def calc_k(r1, r2):
        if abs(r1 - r2) < 1e-8:
            return 1 / (1 + r1)
        return (np.log(1 + r1) - np.log(1 + r2)) / (r1 - r2)
            
    K_CA = calc_k(C_cum, A_cum)
    k_CA_t = df_res.apply(lambda row: calc_k(row['ベンチマーク(C)'], row['目標(A)']), axis=1)
    coef_CA = k_CA_t / K_CA
    
    K_BC = calc_k(B_cum, C_cum)
    k_BC_t = df_res.apply(lambda row: calc_k(row['実績(B)'], row['ベンチマーク(C)']), axis=1)
    coef_BC = k_BC_t / K_BC
    
    res_cum = {
        '累積目標(A)': A_cum, '累積ベンチマーク(C)': C_cum, '累積実績(B)': B_cum,
        '累積マクロ環境(C-A)': C_cum - A_cum, '累積行動(B-C)': B_cum - C_cum,
        '累積環境_金利': (df_res['環境_金利'] * coef_CA).sum(),
        '累積環境_株式等': (df_res['環境_株式等'] * coef_CA).sum(),
        '累積環境_目標控除': (df_res['環境_目標控除'] * coef_CA).sum(),
        '累積環境_誤差': (df_res['環境_誤差'] * coef_CA).sum(),
        '累積行動_配分': (df_res['行動_配分'] * coef_BC).sum(),
        '累積行動_銘柄': (df_res['行動_銘柄'] * coef_BC).sum(),
        '累積行動_売買': (df_res['行動_売買'] * coef_BC).sum(),
        '累積行動_コスト': (df_res['行動_コスト'] * coef_BC).sum(),
        '累積行動_誤差': (df_res['行動_誤差'] * coef_BC).sum()
    }
    
    for ast in attr_list:
        res_cum[f'累積配分_{ast}'] = (df_res[f'配分_{ast}'] * coef_BC).sum()
        res_cum[f'累積銘柄_{ast}'] = (df_res[f'銘柄_{ast}'] * coef_BC).sum()
        res_cum[f'累積売買_{ast}'] = (df_res[f'売買_{ast}'] * coef_BC).sum()
        res_cum[f'累積コスト_{ast}'] = (df_res[f'コスト_{ast}'] * coef_BC).sum()
        res_cum[f'累積誤差_{ast}'] = (df_res[f'誤差_{ast}'] * coef_BC).sum()
        
    return res_cum

asset_classes = df_bm_attr['資産区分'].tolist()
cum_res = calculate_cumulative_carino(df_results, asset_classes)

# =========================================================
# 4. 可視化（ストーリー性のあるグラフ群）
# =========================================================
# グラフ①～④：縦幅を狭め、文字サイズを拡大、半透明を解消
def plot_waterfall(ax, categories, values, bottoms, colors, title, ylabel=''):
    ax.bar(categories, values, bottom=bottoms, color=colors, edgecolor='white', alpha=1.0)
    ax.set_title(title, fontsize=16, fontweight='bold')
    if ylabel: ax.set_ylabel(ylabel, fontsize=14)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.tick_params(axis='both', which='major', labelsize=12)
    for i, v in enumerate(values):
        text_color = 'white' if colors[i] in ['navy', 'dimgray', 'purple', 'gray', '#4472c4'] else 'black'
        ax.text(i, bottoms[i] + v/2, f"{v:+.2f}%", ha='center', va='center', color=text_color, fontweight='bold', fontsize=12)

def get_bot_col(values, pos_c, neg_c, tot_pos_c, tot_neg_c):
    bot = [0]
    curr = 0
    cols = []
    for v in values[:-1]:
        cols.append(pos_c if v >= 0 else neg_c)
        curr += v
        bot.append(curr)
    bot[-1] = 0
    cols.append(tot_pos_c if values[-1] >= 0 else tot_neg_c)
    return bot, cols

# グラフサイズを (18, 12) から (18, 10) へ変更し長方形寄りに
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.2)
ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

cat1 = ['累積目標(A)', '環境要因(C-A)', '行動要因(B-C)', '累積実績(B)']
val1 = [cum_res['累積目標(A)']*100, cum_res['累積マクロ環境(C-A)']*100, cum_res['累積行動(B-C)']*100, cum_res['累積実績(B)']*100]
bot1 = [0, val1[0], val1[0] + val1[1], 0]
col1 = ['gray', 'skyblue' if val1[1] >= 0 else 'salmon', 'lightgreen' if val1[2] >= 0 else 'salmon', 'navy']
plot_waterfall(ax1, cat1, val1, bot1, col1, '① マクロ・アトリビューション (累積期間)', '累積リターン (%)')

cat2 = ['金利', '株式等', '誤差', '目標控除', '全体(C-A)']
val2 = [cum_res['累積環境_金利']*100, cum_res['累積環境_株式等']*100, cum_res['累積環境_誤差']*100, cum_res['累積環境_目標控除']*100, cum_res['累積マクロ環境(C-A)']*100]
bot2, col2 = get_bot_col(val2, 'lightblue', 'lightpink', 'skyblue', 'salmon')
plot_waterfall(ax2, cat2, val2, bot2, col2, '② 環境要因(C-A)の累積詳細', '寄与度 (%)')

cat3 = ['配分', '銘柄', '売買', 'コスト', '誤差', '全体(B-C)']
val3 = [cum_res['累積行動_配分']*100, cum_res['累積行動_銘柄']*100, cum_res['累積行動_売買']*100, cum_res['累積行動_コスト']*100, cum_res['累積行動_誤差']*100, cum_res['累積行動(B-C)']*100]
bot3, col3 = get_bot_col(val3, 'palegreen', 'lightpink', 'lightgreen', 'salmon')
plot_waterfall(ax3, cat3, val3, bot3, col3, '③ 行動要因(B-C)の全体詳細', '寄与度 (%)')

factors = ['配分', '銘柄', '売買', 'コスト', '誤差']
x = np.arange(len(factors))
width = 0.6
pos_bottom = np.zeros(len(factors))
neg_bottom = np.zeros(len(factors))

colors_palette = ['#5b9bd5', '#9dc3e6', '#ed7d31', '#f4b084', '#a5a5a5', '#ffc000', '#70ad47', '#4472c4']

for i, ast in enumerate(asset_classes):
    vals = np.array([
        cum_res[f'累積配分_{ast}'] * 100,
        cum_res[f'累積銘柄_{ast}'] * 100,
        cum_res[f'累積売買_{ast}'] * 100,
        cum_res[f'累積コスト_{ast}'] * 100,
        cum_res[f'累積誤差_{ast}'] * 100
    ])
    
    pos_vals = np.maximum(vals, 0)
    neg_vals = np.minimum(vals, 0)
    
    color = colors_palette[i % len(colors_palette)]
    ax4.bar(x, pos_vals, width, bottom=pos_bottom, color=color, edgecolor='white', alpha=0.5, label=ast)
    ax4.bar(x, neg_vals, width, bottom=neg_bottom, color=color, edgecolor='white', alpha=0.7)
    
    pos_bottom += pos_vals
    neg_bottom += neg_vals

ax4.set_xticks(x)
ax4.set_xticklabels(factors, fontsize=12)
ax4.tick_params(axis='y', labelsize=12)
ax4.set_title('④ 各行動要因の資産区分別内訳（8資産）', fontsize=16, fontweight='bold')
ax4.set_ylabel('寄与度 (%)', fontsize=14)
ax4.axhline(0, color='black', linewidth=0.8)

# ④に要因別の合計値（ネット寄与度）を黒い点でプロットし、データラベルをプロットの上下に配置
totals = np.array([cum_res[f'累積行動_{f}'] * 100 for f in factors])
ax4.plot(x, totals, marker='o', color='black', linestyle='None', markersize=6, zorder=5, label='合計')

for i, tot in enumerate(totals):
    # annotateを使用して、プロットした点の少し上（プラス）か少し下（マイナス）にラベルを表示
    ax4.annotate(f"{tot:+.2f}%", (x[i], tot), textcoords="offset points", xytext=(0, 10 if tot >= 0 else -15), ha='center', fontweight='bold', color='black', fontsize=12)

handles, labels = ax4.get_legend_handles_labels()
# プロットした「合計」が凡例の上部にくるよう調整
ax4.legend(handles[::-1], labels[::-1], loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=12)

plt.tight_layout()
plt.show()

# --- グラフ群2：時系列推移（⑤～⑦） ---
fig_ts, (ax_ts1, ax_ts2, ax_ts3) = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [1, 1, 1.2]})
plt.subplots_adjust(hspace=0.4)
x_labels = df_results['年月']

cum_A_series = (1 + df_results['目標(A)']).cumprod() - 1
cum_B_series = (1 + df_results['実績(B)']).cumprod() - 1
cum_diff = cum_B_series - cum_A_series

ax_ts1.plot(x_labels, cum_B_series * 100, marker='o', color='navy', label='累積実績 (B)', linewidth=2)
ax_ts1.plot(x_labels, cum_A_series * 100, marker='s', color='gray', label='累積目標 (A)', linewidth=2, linestyle='--')
colors_diff_cum = ['lightgreen' if x >= 0 else 'salmon' for x in cum_diff]
ax_ts1.bar(x_labels, cum_diff * 100, alpha=0.5, color=colors_diff_cum, label='累積目標超過 (B-A)')
ax_ts1.axhline(0, color='black', linewidth=1)
ax_ts1.set_title('⑤ 累積リターンの推移と目標超過額 (実績 vs 目標)', fontsize=15, fontweight='bold')
ax_ts1.legend(loc='upper left')

# データラベルを 実績(B) に変更し、色をネイビーに
for i, val in enumerate(cum_B_series):
    ax_ts1.annotate(f"{val*100:+.2f}%", (x_labels.iloc[i], val*100), textcoords="offset points", xytext=(0,10 if val>=0 else -15), ha='center', fontsize=10, fontweight='bold', color='navy')

m_diff = df_results['実績(B)'] - df_results['目標(A)']
ax_ts2.plot(x_labels, df_results['実績(B)'] * 100, marker='o', color='navy', label='月次実績 (B)', linewidth=2)
ax_ts2.plot(x_labels, df_results['目標(A)'] * 100, marker='s', color='gray', label='月次目標 (A)', linewidth=2, linestyle='--')
colors_diff_m = ['lightgreen' if x >= 0 else 'salmon' for x in m_diff]
ax_ts2.bar(x_labels, m_diff * 100, alpha=0.5, color=colors_diff_m, label='月次目標超過 (B-A)')
ax_ts2.axhline(0, color='black', linewidth=1)
ax_ts2.set_title('⑥ 月次リターンの推移と目標超過額 (実績 vs 目標)', fontsize=15, fontweight='bold')
ax_ts2.legend(loc='upper left')

# データラベルを 実績(B) に変更し、色をネイビーに
for i, val in enumerate(df_results['実績(B)']):
    ax_ts2.annotate(f"{val*100:+.2f}%", (x_labels.iloc[i], val*100), textcoords="offset points", xytext=(0,10 if val>=0 else -15), ha='center', fontsize=10, fontweight='bold', color='navy')

env_series = df_results['マクロ環境(C-A)'] * 100
skill_series = df_results['行動(B-C)'] * 100
total_excess = (df_results['実績(B)'] - df_results['目標(A)']) * 100
ax_ts3.bar(x_labels, env_series, label='環境要因 (C-A)', color='skyblue', edgecolor='white')
ax_ts3.bar(x_labels, np.maximum(skill_series, 0), bottom=np.maximum(env_series, 0), label='行動要因 (B-C) プラス', color='lightgreen', edgecolor='white')
ax_ts3.bar(x_labels, np.minimum(skill_series, 0), bottom=np.minimum(env_series, 0), label='行動要因 (B-C) マイナス', color='salmon', edgecolor='white')

# ⑦に月次目標超過(B-A)をプロットとして追加（凡例用・線なし・サイズ6）
ax_ts3.plot(x_labels, total_excess, marker='o', color='black', linestyle='None', label='月次目標超過 (B-A)', markersize=6, zorder=5)

ax_ts3.axhline(0, color='black', linewidth=1)
ax_ts3.set_title('⑦ 【月次】目標超過(B-A)の要因分解推移（内訳のみ）', fontsize=15, fontweight='bold')
ax_ts3.legend(loc='upper left')

# ⑦のデータラベルは今まで通り超過リターンを黒字で出力
for i, val in enumerate(total_excess):
    ax_ts3.annotate(f"{val:+.2f}%", (x_labels.iloc[i], val), textcoords="offset points", xytext=(0,10 if val>=0 else -15), ha='center', fontweight='bold', color='black')

plt.tight_layout()
plt.show()

# =========================================================
# 5. エクセルファイルへの自動出力（分析結果）
# =========================================================
print("\n=== Excelファイルへの出力処理を開始します ===")
output_file = 'Performance_Attribution_Result.xlsx'

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    df_macro.to_excel(writer, sheet_name='1_Input_マクロ指標', index=False)
    df_master.to_excel(writer, sheet_name='1_Input_銘柄マスタ', index=False)
    df_bm_attr.to_excel(writer, sheet_name='1_Input_BM設定', index=False)
    df_holdings.to_excel(writer, sheet_name='1_Input_月末残高', index=False)
    df_transactions.to_excel(writer, sheet_name='1_Input_期中取引', index=False)
    
    df_results.to_excel(writer, sheet_name='2_Calc_月次要因分解', index=False)
    df_cum_res = pd.DataFrame([cum_res]).T.reset_index()
    df_cum_res.columns = ['項目', '値']
    df_cum_res.to_excel(writer, sheet_name='2_Calc_累積要因分解', index=False)
    
    df_graph_ts = df_results[['年月', '目標(A)', '実績(B)', 'マクロ環境(C-A)', '行動(B-C)']].copy()
    df_graph_ts['月次_超過(B-A)'] = df_graph_ts['実績(B)'] - df_graph_ts['目標(A)']
    df_graph_ts['累積_目標(A)'] = (1 + df_graph_ts['目標(A)']).cumprod() - 1
    df_graph_ts['累積_実績(B)'] = (1 + df_graph_ts['実績(B)']).cumprod() - 1
    df_graph_ts['累積_超過(B-A)'] = df_graph_ts['累積_実績(B)'] - df_graph_ts['累積_目標(A)']
    
    base_cols = ['環境_金利', '環境_株式等', '環境_誤差', '環境_目標控除', '行動_配分', '行動_銘柄', '行動_売買', '行動_コスト', '行動_誤差']
    df_graph_ts = pd.concat([df_graph_ts, df_results[base_cols]], axis=1)
    
    asset_cols = [c for c in df_results.columns if any(c.startswith(prefix) for prefix in ['配分_', '銘柄_', '売買_', 'コスト_', '誤差_'])]
    df_graph_ts = pd.concat([df_graph_ts, df_results[asset_cols]], axis=1)
    
    df_graph_ts.to_excel(writer, sheet_name='3_Result_グラフ用データ', index=False)
    
    workbook  = writer.book
    pct_fmt = workbook.add_format({'num_format': '0.00%'})
    
    for sheet_name in writer.sheets:
        worksheet = writer.sheets[sheet_name]
        if sheet_name.startswith('2_') or sheet_name.startswith('3_'):
            worksheet.set_column('B:AZ', 12, pct_fmt)
        elif sheet_name == '1_Input_マクロ指標':
            worksheet.set_column('B:Z', 12, pct_fmt)

print(f"分析結果Excelファイル '{output_file}' を出力しました。")


# =========================================================
# 6. 将来の目標達成確率シミュレーション（本格的共分散モデル）
# =========================================================
print("\n=== 将来シミュレーション（目標達成確率）の生成を開始します ===")

past_months = 12
future_months = 24
total_months = past_months + future_months

past_B_cum = (1 + df_results['実績(B)']).cumprod().values[-1] - 1
past_A_cum = (1 + df_results['目標(A)']).cumprod().values[-1] - 1

# ---------------------------------------------------------
# (A) 将来の政策金利パスと目標リターンの設定
# ---------------------------------------------------------
# 2026年3月末(現在):0.75%
# 2026年6月末(+3ヶ月後):1.00%, 2026年12月末(+9ヶ月後):1.25%, 2027年6月末(+15ヶ月後):1.50%
future_policy_rates = []
current_pr = 0.0075
for i in range(1, future_months + 1):
    if i == 3:  current_pr = 0.0100  # 2026年6月末 (15ヶ月目)
    if i == 9:  current_pr = 0.0125  # 2026年12月末 (21ヶ月目)
    if i == 15: current_pr = 0.0150  # 2027年6月末 (27ヶ月目)
    future_policy_rates.append(current_pr)

# 月次目標リターン = (政策金利 + 80bps) / 12
future_A_returns = [(pr + 0.0080) / 12 for pr in future_policy_rates]
future_A_cum_path = [past_A_cum]
current_A_cum = past_A_cum
for ret in future_A_returns:
    current_A_cum = (1 + current_A_cum) * (1 + ret) - 1
    future_A_cum_path.append(current_A_cum)

# ---------------------------------------------------------
# (B) 各資産クラスの前提条件（期待リターン・ボラティリティ・相関行列）
# ---------------------------------------------------------
asset_assumptions = pd.DataFrame([
    {'資産区分': '内債_固定',    'mu': -0.015, 'vol': 0.030}, # 金利上昇で価格下落
    {'資産区分': '内債_変動',    'mu': 0.015,  'vol': 0.010}, # 金利上昇に追随
    {'資産区分': '外債H有_固定', 'mu': 0.005,  'vol': 0.040},
    {'資産区分': '外債H有_変動', 'mu': 0.025,  'vol': 0.020},
    {'資産区分': '外債Hなし',    'mu': 0.035,  'vol': 0.100},
    {'資産区分': '内株',         'mu': 0.060,  'vol': 0.150},
    {'資産区分': 'J-REIT',       'mu': 0.050,  'vol': 0.120},
    {'資産区分': '投信',         'mu': 0.040,  'vol': 0.080}
]).set_index('資産区分')

corr_matrix = pd.DataFrame(
    [[ 1.0,  0.8,  0.6,  0.4,  0.2, -0.1,  0.1,  0.3],
     [ 0.8,  1.0,  0.5,  0.6,  0.2, -0.1,  0.1,  0.3],
     [ 0.6,  0.5,  1.0,  0.8,  0.4, -0.2,  0.0,  0.2],
     [ 0.4,  0.6,  0.8,  1.0,  0.4, -0.2,  0.0,  0.2],
     [ 0.2,  0.2,  0.4,  0.4,  1.0,  0.4,  0.3,  0.5],
     [-0.1, -0.1, -0.2, -0.2,  0.4,  1.0,  0.6,  0.8],
     [ 0.1,  0.1,  0.0,  0.0,  0.3,  0.6,  1.0,  0.6],
     [ 0.3,  0.3,  0.2,  0.2,  0.5,  0.8,  0.6,  1.0]],
    index=asset_classes, columns=asset_classes
)

# ---------------------------------------------------------
# (C) 将来の行動シナリオ（ポートフォリオ・トランジション）
# ---------------------------------------------------------
w_base_dict = {'内債_固定': 0.30, '内債_変動': 0.10, '外債H有_固定': 0.10, '外債H有_変動': 0.05, '外債Hなし': 0.05, '内株': 0.15, 'J-REIT': 0.05, '投信': 0.20}
w_base = np.array([w_base_dict[ast] for ast in asset_classes])

w_act_dict = {'内債_固定': 0.15, '内債_変動': 0.20, '外債H有_固定': 0.10, '外債H有_変動': 0.05, '外債Hなし': 0.05, '内株': 0.20, 'J-REIT': 0.05, '投信': 0.20}
w_act = np.array([w_act_dict[ast] for ast in asset_classes])
transition_months = 12

# ---------------------------------------------------------
# (D) モンテカルロ・シミュレーションの実行（同一相場環境下での比較）
# ---------------------------------------------------------
n_sims = 5000
np.random.seed(42)

L = np.linalg.cholesky(corr_matrix.values)

mu_assets_m = asset_assumptions['mu'].values / 12
vol_assets_m = asset_assumptions['vol'].values / np.sqrt(12)

cum_sim_base = np.zeros((n_sims, future_months + 1))
cum_sim_base[:, 0] = past_B_cum
cum_sim_act = np.zeros((n_sims, future_months + 1))
cum_sim_act[:, 0] = past_B_cum

for i in range(1, future_months + 1):
    Z_uncorr = np.random.normal(0, 1, (len(asset_classes), n_sims))
    Z_corr = np.dot(L, Z_uncorr)
    
    asset_ret_sim = mu_assets_m[:, np.newaxis] + vol_assets_m[:, np.newaxis] * Z_corr
    
    progress = min(i / transition_months, 1.0)
    w_t = w_base + (w_act - w_base) * progress
    
    port_ret_base = np.dot(w_base, asset_ret_sim)
    port_ret_act = np.dot(w_t, asset_ret_sim)
    
    cum_sim_base[:, i] = (1 + cum_sim_base[:, i-1]) * (1 + port_ret_base) - 1
    cum_sim_act[:, i] = (1 + cum_sim_act[:, i-1]) * (1 + port_ret_act) - 1

target_final = future_A_cum_path[-1]
prob_base = np.sum(cum_sim_base[:, -1] >= target_final) / n_sims
prob_act = np.sum(cum_sim_act[:, -1] >= target_final) / n_sims

# ---------------------------------------------------------
# (E) グラフ描画（X軸を日付化、利上げマーカー追加）
# ---------------------------------------------------------
fig_sim, ax_sim = plt.subplots(figsize=(12, 7))

# freq='M' から freq='ME' へ修正
sim_dates = pd.date_range(start='2025-03-31', periods=total_months + 1, freq='ME')
x_past = sim_dates[:past_months + 1]
x_future = sim_dates[past_months:]

past_A_path = np.concatenate(([0], (1 + df_results['目標(A)']).cumprod().values - 1))
past_B_path = np.concatenate(([0], (1 + df_results['実績(B)']).cumprod().values - 1))

ax_sim.plot(x_past, past_A_path * 100, color='gray', linestyle='--', linewidth=2, label='過去の累積目標(A)')
ax_sim.plot(x_past, past_B_path * 100, color='navy', linewidth=2, marker='o', label='過去の累積実績(B)')
ax_sim.plot(x_future, np.array(future_A_cum_path) * 100, color='black', linestyle='--', linewidth=2, label='将来の目標累積パス')

# 利上げタイミングのマーカー追加 (3ヶ月後:2026/06, 9ヶ月後:2026/12, 15ヶ月後:2027/06)
hike_idx = [3, 9, 15]
hike_dates = [x_future[i] for i in hike_idx]
hike_vals = [future_A_cum_path[i] * 100 for i in hike_idx]
ax_sim.scatter(hike_dates, hike_vals, color='red', marker='^', s=150, zorder=5, label='利上げ想定タイミング')

# ファンチャート(現状)
p10_base = np.percentile(cum_sim_base, 10, axis=0) * 100
p90_base = np.percentile(cum_sim_base, 90, axis=0) * 100
p50_base = np.percentile(cum_sim_base, 50, axis=0) * 100
ax_sim.fill_between(x_future, p10_base, p90_base, color='lightsteelblue', alpha=0.4, label='現状維持 80%確率区間')
ax_sim.plot(x_future, p50_base, color='royalblue', linestyle=':', linewidth=2, label='現状維持 中央値')

# ファンチャート(行動後)
p10_act = np.percentile(cum_sim_act, 10, axis=0) * 100
p90_act = np.percentile(cum_sim_act, 90, axis=0) * 100
p50_act = np.percentile(cum_sim_act, 50, axis=0) * 100
ax_sim.fill_between(x_future, p10_act, p90_act, color='lightgreen', alpha=0.5, label='行動後 80%確率区間')
ax_sim.plot(x_future, p50_act, color='forestgreen', linestyle='-', linewidth=2, label='行動後 中央値')

# X軸のフォーマット (日付表示)
ax_sim.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax_sim.xaxis.get_majorticklabels(), rotation=45, ha='right')

# メッセージボックス
textstr = (
    f"【3年後の目標達成確率】\n"
    f"・現状維持 : {prob_base*100:.1f}%\n"
    f"・行動後  : {prob_act*100:.1f}%"
    # f"《シミュレーションの戦略前提》\n"
    # f"政策金利は2026年6月・12月、2027年6月に25bpsの利上げを想定。\n"
    # f"金利上昇による固定債の価格下落リスクを回避するため、\n"
    # f"「内債_固定」を15%削減し、「内債_変動(+10%)」と\n"
    # f"「内株(+5%)」へ、今後12ヶ月かけて段階的にシフト。"
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='gray')
ax_sim.text(0.5, 0.95, textstr, transform=ax_sim.transAxes, fontsize=12, verticalalignment='top', bbox=props, fontweight='bold')

# 過去と未来の境界線
ax_sim.axvline(sim_dates[past_months], color='red', linestyle='-', alpha=0.7)
ax_sim.text(sim_dates[past_months] + pd.Timedelta(days=10), ax_sim.get_ylim()[0] + 0.5, '現在 (1年経過)', color='red', rotation=90, verticalalignment='bottom', fontweight='bold', fontsize=12)

ax_sim.set_title('⑧ 将来シミュレーション：戦略的行動による「目標達成確率」の変化', fontsize=16, fontweight='bold')
ax_sim.set_xlabel('年月', fontsize=14)
ax_sim.set_ylabel('累積リターン (%)', fontsize=14)
ax_sim.tick_params(axis='both', which='major', labelsize=12)

ax_sim.legend(loc='upper left', fontsize=11)

plt.tight_layout()
plt.show()

# =========================================================
# 7. シミュレーション用 Excelファイルへの自動出力
# =========================================================
print("\n=== シミュレーション結果Excelファイルへの出力処理を開始します ===")
sim_output_file = 'Simulation_Result.xlsx'

with pd.ExcelWriter(sim_output_file, engine='xlsxwriter') as writer:
    # 1. 政策金利・目標設定パス
    future_months_labels = [(pd.to_datetime('2026-03-31') + pd.DateOffset(months=i)).strftime('%Y-%m') for i in range(1, future_months + 1)]
    df_sim_target = pd.DataFrame({
        '経過月数': range(13, 37),
        '年月': future_months_labels,
        '政策金利': future_policy_rates,
        '目標リターン_月次': future_A_returns,
        '累積目標パス_A': future_A_cum_path[1:]
    })
    df_sim_target.to_excel(writer, sheet_name='1_Target_Path', index=False)

    # 2. 資産ごとのリスク・リターン前提
    asset_assumptions.reset_index().to_excel(writer, sheet_name='2_Asset_Assumptions', index=False)
    
    # 3. 相関行列
    corr_matrix.reset_index().rename(columns={'index': '資産区分'}).to_excel(writer, sheet_name='3_Correlation_Matrix', index=False)
    
    # 4. 行動の前提（ウェイト変化）
    df_weights = pd.DataFrame({
        '資産区分': asset_classes,
        '現状維持_Weight': w_base,
        '行動後(目標)_Weight': w_act
    })
    df_weights.to_excel(writer, sheet_name='4_Weights', index=False)

    # 5. シミュレーションサマリー（確率とパーセンタイル）
    df_sim_summary = pd.DataFrame({
        'シナリオ': ['現状維持(Do Nothing)', '行動後(Action)'],
        '目標達成確率': [prob_base, prob_act],
        '3年後リターン_10%ile (悲観)': [p10_base[-1]/100, p10_act[-1]/100],
        '3年後リターン_50%ile (中位)': [p50_base[-1]/100, p50_act[-1]/100],
        '3年後リターン_90%ile (楽観)': [p90_base[-1]/100, p90_act[-1]/100],
    })
    df_sim_summary.to_excel(writer, sheet_name='5_Summary', index=False)

    # 6. シミュレーション全パス出力（各シナリオ 先頭1000パスを書き出し）
    save_paths = 1000
    df_paths_base = pd.DataFrame(cum_sim_base[:save_paths, 1:], columns=future_months_labels)
    df_paths_base.insert(0, 'Path_ID', range(1, save_paths + 1))
    df_paths_base.to_excel(writer, sheet_name='6_Paths_Base', index=False)

    df_paths_act = pd.DataFrame(cum_sim_act[:save_paths, 1:], columns=future_months_labels)
    df_paths_act.insert(0, 'Path_ID', range(1, save_paths + 1))
    df_paths_act.to_excel(writer, sheet_name='7_Paths_Act', index=False)

    # 書式設定
    workbook  = writer.book
    pct_fmt = workbook.add_format({'num_format': '0.00%'})
    for sheet_name in writer.sheets:
        worksheet = writer.sheets[sheet_name]
        if sheet_name == '1_Target_Path':
            worksheet.set_column('C:E', 12, pct_fmt)
        elif sheet_name == '2_Asset_Assumptions':
            worksheet.set_column('B:C', 12, pct_fmt)
        elif sheet_name == '4_Weights':
            worksheet.set_column('B:C', 12, pct_fmt)
        elif sheet_name == '5_Summary':
            worksheet.set_column('B:E', 15, pct_fmt)
        elif sheet_name in ['6_Paths_Base', '7_Paths_Act']:
            worksheet.set_column('B:Y', 10, pct_fmt)

print(f"シミュレーション結果Excelファイル '{sim_output_file}' を出力しました。")
print("入力設定、相関行列、および乱数パスの生データが格納されています。")
