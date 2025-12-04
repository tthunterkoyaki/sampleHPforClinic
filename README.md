# --------------------------------------------------------------------------------
# 7. 追加機能：シナリオ作成と詳細可視化 (Scenario & Visualization Extension)
# --------------------------------------------------------------------------------

def create_linear_scenario_df(start_date, end_date, start_rates, end_rates):
    """
    指定した開始・終了レート間を直線で結ぶシナリオデータを作成する
    start_rates, end_rates: {'USDJPY': 150, 'EURJPY': 160} のような辞書
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    data = {}
    
    for pair, start_val in start_rates.items():
        end_val = end_rates.get(pair, start_val) # 指定がなければ横ばい
        # linspaceで等間隔の数値を生成
        data[pair] = np.linspace(start_val, end_val, len(dates))
        
    return pd.DataFrame(data, index=dates)

def create_custom_scenario_df(dates, rates_dict):
    """
    任意の日付とレート配列からシナリオデータを作成する
    rates_dict: {'USDJPY': [150, 151, ...], ...}
    """
    df = pd.DataFrame(rates_dict, index=pd.to_datetime(dates))
    # 営業日ベースにリサンプリングして欠損を埋める（判定ロジックが営業日連続を前提とする場合があるため）
    df = df.resample('B').ffill()
    return df

def visualize_contract_detail(contract, fx_scenario_df, ko_date=None):
    """
    【可視化メイン】
    特定の契約と為替シナリオを受け取り、判定結果をチャートに描画する。
    なぜそこで消滅したのか、あるいは生き残ったのかを視覚的に説明する。
    """
    import matplotlib.dates as mdates

    # --- 1. 描画データの準備 ---
    pair = 'USDJPY' if 'USD' in [contract['受取通貨'], contract['支払通貨']] else \
           ('EURJPY' if 'EUR' in [contract['受取通貨'], contract['支払通貨']] else 'CNHJPY')
    
    if pair not in fx_scenario_df.columns:
        print(f"Error: シナリオデータに {pair} が含まれていません。")
        return

    # 期間: 契約日の少し前から、終了日（または消滅日）の少し後まで
    view_start = contract['契約日'] - timedelta(days=30)
    view_end = contract['終了日'] + timedelta(days=30)
    
    # シナリオデータの切り出し
    # plot_data = fx_scenario_df[pair].loc[view_start:view_end] # インデックスエラー回避のため全体を使う
    plot_data = fx_scenario_df[pair]
    # ただし描画範囲はset_xlimで制限する

    # --- 2. プロット作成 ---
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 為替レートの推移
    ax.plot(plot_data.index, plot_data.values, label=f'{pair} Rate', color='#333333', linewidth=1.5)

    # --- 3. 契約情報の描画 (水平線・垂直線) ---
    
    # 契約期間のシェーディング（背景色）
    effective_end = ko_date if ko_date else contract['終了日']
    ax.axvspan(contract['契約日'], effective_end, color='eef', alpha=0.3, label='有効期間')
    
    # 重要な日付のライン
    ax.axvline(contract['契約日'], color='gray', linestyle=':', alpha=0.8)
    ax.text(contract['契約日'], ax.get_ylim()[1], '契約開始', rotation=90, va='top', ha='right', fontsize=9)
    
    ax.axvline(contract['終了日'], color='gray', linestyle=':', alpha=0.8)
    ax.text(contract['終了日'], ax.get_ylim()[1], '満期', rotation=90, va='top', ha='left', fontsize=9)

    # --- 4. デリバティブタイプ別の描画ロジック ---
    
    # A. 通常KO型 (バリアの描画)
    if contract['通常KO型'] == 1:
        ko_rate = contract['KO・KI相場(1行目)']
        ax.axhline(ko_rate, color='red', linestyle='-', linewidth=2, label=f'KO Barrier: {ko_rate:.2f}')
        
        # 判定タイプ
        judge_type = contract['KO・KI判定(1行目)']
        ax.text(view_start, ko_rate, f' KOバリア ({judge_type})', color='red', va='bottom', fontsize=10, fontweight='bold')
        
        # ヨーロピアンの場合、交換日（判定日）をプロット
        if judge_type == 'ﾖｰﾛﾋﾟｱﾝ':
            ex_dates = get_exchange_dates(contract['初回決済日'], contract['終了日'], contract['年間交換回数'], contract['交換日'])
            # 期間内の交換日のみ
            valid_ex_dates = [d for d in ex_dates if contract['契約日'] <= d <= contract['終了日']]
            # その日のレートを取得
            ex_rates = [get_rate_at(fx_scenario_df, d, pair) for d in valid_ex_dates]
            ax.scatter(valid_ex_dates, ex_rates, color='orange', s=50, marker='o', zorder=5, label='判定日(交換日)')

    # B. ターゲットKO型 (ターゲットレートの描画)
    elif contract['ターゲットKO型'] == 1:
        rate1 = contract['決済相場1']
        rate2 = contract['決済相場2']
        switch_count = contract['支払件数(特殊形の抽出後)']
        
        # 交換日を取得
        ex_dates = get_exchange_dates(contract['初回決済日'], contract['終了日'], contract['年間交換回数'], contract['交換日'])
        valid_ex_dates = [d for d in ex_dates if contract['契約日'] <= d <= contract['終了日']]
        
        # ターゲットレートの推移を作成（階段状）
        target_series = []
        for i, d in enumerate(valid_ex_dates):
            # iは0始まりなので、i+1回目
            current_target = rate2 if (pd.notna(rate2) and pd.notna(switch_count) and (i+1) > switch_count) else rate1
            target_series.append({'Date': d, 'Target': current_target})
            
        if target_series:
            ts_df = pd.DataFrame(target_series)
            # ターゲットレートをステッププロットで描画（視覚的にわかりやすく）
            ax.step(ts_df['Date'], ts_df['Target'], where='post', color='green', linestyle='--', linewidth=2, label='Target Rate')
            
            # 交換日をプロット
            ex_rates = [get_rate_at(fx_scenario_df, d, pair) for d in valid_ex_dates]
            ax.scatter(valid_ex_dates, ex_rates, color='green', s=40, marker='D', zorder=5, label='交換日')

    # --- 5. KO発生時の強調 ---
    if ko_date:
        ko_val = get_rate_at(fx_scenario_df, ko_date, pair)
        ax.scatter(ko_date, ko_val, color='red', s=200, marker='X', zorder=10, label='KO発生')
        
        # 吹き出し注釈
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=2)
        ax.annotate(f'Knock Out!\n{ko_date.strftime("%Y-%m-%d")}\nRate: {ko_val:.2f}', 
                    xy=(ko_date, ko_val), xytext=(20, 30), textcoords='offset points',
                    bbox=bbox_props, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    else:
        # 満期到達の注釈
        last_val = get_rate_at(fx_scenario_df, contract['終了日'], pair)
        if pd.notna(last_val):
            ax.annotate('満期到達', xy=(contract['終了日'], last_val), xytext=(20, 20), textcoords='offset points',
                        bbox=dict(boxstyle="round", fc="white", ec="green"), arrowprops=dict(arrowstyle="->"))

    # --- 6. グラフの整形 ---
    ax.set_title(f"契約詳細シミュレーション: {contract['契約ID']} ({contract['_Type']}) / {contract['受取通貨']}受取", fontsize=14)
    ax.set_ylabel(f'{pair} Rate')
    ax.set_xlim(view_start, view_end)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.grid(True, which='major', linestyle='--', alpha=0.7)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()

    # --- 7. 数値結果の出力（テキスト） ---
    print(f"--- Contract: {contract['契約ID']} Simulation Result ---")
    print(f"Type: {contract['_Type']}")
    if ko_date:
        print(f"Result: KO発生 (Date: {ko_date.strftime('%Y-%m-%d')})")
        # どの条件で消滅したか簡易判定
        if contract['通常KO型'] == 1:
            print(f"Reason: レートがバリア({contract['KO・KI相場(1行目)']})に到達")
        elif contract['ターゲットKO型'] == 1:
            tgt_info = f"Amount >= {contract['ターゲット金額（円）']}" if pd.notna(contract['ターゲット金額（円）']) else f"Count >= {contract['ターゲット回数（件数）']}"
            print(f"Reason: 蓄積条件達成 ({tgt_info})")
    else:
        print(f"Result: 満期到達 (Date: {contract['終了日'].strftime('%Y-%m-%d')})")


# --------------------------------------------------------------------------------
# 実行ブロック：追加分析の使用例
# --------------------------------------------------------------------------------

if __name__ == '__main__':
    # 前提: contracts_df が既に生成されていること
    
    print("\n" + "="*60)
    print("【追加分析】 指定シナリオによる詳細検証")
    print("="*60)

    # ----------------------------------------------
    # Step 1: 分析対象の契約を選ぶ（例としてランダムに1つ）
    # ----------------------------------------------
    # 例: まだ消滅していない、かつKO条件がある契約を探す
    target_candidates = contracts_df[
        contracts_df['消滅日'].isna() & 
        ((contracts_df['通常KO型'] == 1) | (contracts_df['ターゲットKO型'] == 1))
    ]
    
    if not target_candidates.empty:
        # ランダムに1つ選択（ID指定も可能: target_contract = contracts_df[contracts_df['契約ID']=='C001'].iloc[0]）
        target_contract = target_candidates.iloc[0] 
        
        print(f"Selected Contract: {target_contract['契約ID']}")
        print(f"Pair: {'USD' if 'USD' in target_contract['受取通貨'] else 'EUR'}JPY (Demo)")
        
        # ----------------------------------------------
        # Step 2: シナリオを作成する
        # ----------------------------------------------
        
        # A. 線形シナリオ（円安進行ケース）
        # 2025/4/1 -> 2026/3/31 で USDJPYが 150 -> 170 に一直線に進む
        scenario_dates = pd.date_range('2023-01-01', '2028-12-31', freq='B') # 期間は広めにとっておく
        
        # ダミーの過去データ部分（一定値で埋める簡易的な例）
        # ※本来は historical_fx_df を結合する方が正確ですが、ここでは挙動確認のためシンプルに作成
        base_rates = {'USDJPY': 150.0, 'EURJPY': 160.0, 'CNHJPY': 20.0}
        
        # シナリオ開始日（分析開始日など）
        sc_start = pd.to_datetime('2025-04-01')
        sc_end = pd.to_datetime('2027-03-31')
        
        # 線形推移データの作成
        linear_future = create_linear_scenario_df(
            sc_start, sc_end, 
            start_rates={'USDJPY': 150.0, 'EURJPY': 160.0}, 
            end_rates={'USDJPY': 170.0, 'EURJPY': 140.0} # USDは円安、EURは円高へ
        )
        
        # 過去データと結合（今回は直近値を過去として引き伸ばす簡易結合）
        # 実際には historical_fx_df を使ってください
        full_scenario_df = pd.concat([historical_fx_df, linear_future]).sort_index().ffill()
        
        # ----------------------------------------------
        # Step 3: 判定と可視化を実行
        # ----------------------------------------------
        
        # この契約が、このシナリオでどうなるか判定
        ko_date_result = evaluate_contract_ko(target_contract, full_scenario_df)
        
        # 結果の可視化
        visualize_contract_detail(target_contract, full_scenario_df, ko_date_result)
        
        
        # ----------------------------------------------
        # (Option) B. カスタム時系列の例（ジグザグな動き）
        # ----------------------------------------------
        print("\n--- Custom Zig-Zag Scenario Case ---")
        
        # 4つのポイントを通るレートを作る
        dates_custom = pd.to_datetime(['2025-04-01', '2025-10-01', '2026-04-01', '2027-03-31'])
        rates_custom_usd = [150, 130, 160, 145] # 乱高下
        
        # scipyの補間など使わず、簡易的に区間をリニアでつなぐ
        custom_segments = []
        for i in range(len(dates_custom)-1):
            seg = create_linear_scenario_df(
                dates_custom[i], dates_custom[i+1],
                {'USDJPY': rates_custom_usd[i]}, {'USDJPY': rates_custom_usd[i+1]}
            )
            custom_segments.append(seg)
        
        custom_future = pd.concat(custom_segments).sort_index()
        # 重複削除
        custom_future = custom_future[~custom_future.index.duplicated(keep='first')]
        
        full_scenario_custom = pd.concat([historical_fx_df, custom_future]).sort_index().ffill()
        
        # 判定
        ko_date_custom = evaluate_contract_ko(target_contract, full_scenario_custom)
        
        # 可視化
        visualize_contract_detail(target_contract, full_scenario_custom, ko_date_custom)
        
    else:
        print("検証可能な契約が見つかりませんでした。データ生成数を増やしてください。")

