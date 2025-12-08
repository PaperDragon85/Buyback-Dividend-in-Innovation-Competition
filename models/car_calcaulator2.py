import pandas as pd
import numpy as np
import os
from WindPy import w
from scipy import stats
import warnings
import datetime

# 忽略 pandas 的一些切片警告
warnings.filterwarnings('ignore')

class FastBatchCARCalculator:
    """
    高效批量 CAR 计算器
    
    设计思路：
    1. 批量获取数据：从 Wind 一次性获取所有公司的股价数据，存储到 ./data/interim/all_stock_prices.csv
    2. fit() 函数：遍历每一行数据，计算回归并生成中间结果文件
       - 根据事件日和相对日期计算绝对日期
       - 从内存中的股价数据表提取收益率
       - 与大盘收益率跑 OLS 回归
       - 计算估计期、事件期的超额收益
       - 结果保存到 ./data/interim/CAR/{股票代码}_{事件日期}_{参数}.csv
       - 如果检测到已计算过的文件，则跳过
    3. calculate_car() 函数：计算指定子区间的 CAR
       - 传入多个区间元组（必须是估计期到窗口期的子集）
       - 从中间文件读取数据计算 CAR 和显著性
       - 输出原表格加上新增的 CAR 列
    
    改进点：
    - 交易日定位：事件日定为公告日当天或之后最近的交易日（标准事件研究做法）
    - 数据索引：正确处理 datetime 对象与 DataFrame 索引的匹配
    - 文件命名：使用事件日期区分同一股票在不同日期的多个事件
    - 本地计算：所有日期偏移在本地完成，极大提升速度
    """
    
    def __init__(self, df, stock_col="证券代码", date_col="BPAmtDy", 
                 estimation_window=(-120, -30), event_window=(-10, 20), # 建议event_window稍微给大一点，计算CAR时取子集即可
                 market_index="000001.SH",
                 date_range=None):  # 新增参数：时间筛选范围 (start_date, end_date)
        """
        初始化
        
        参数:
        df: DataFrame, 包含股票和公告日的表格
        stock_col: str, 股票代码列名
        date_col: str, 日期列名
        estimation_window: tuple, 估计期 (相对于 T0)
        event_window: tuple, 最大的事件窗口期 (此范围内的所有数据都会被计算并存储)
        market_index: str, 市场基准
        date_range: tuple, 可选，时间筛选范围 (start_date, end_date)，格式可以是字符串或datetime对象
                   例如: ('2020-01-01', '2023-12-31') 只分析这个时间范围内的事件
        """
        self.df = df.copy()
        
        # 时间筛选
        if date_range is not None:
            start_date, end_date = date_range
            # 转换为datetime
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # 筛选时间范围
            original_len = len(self.df)
            self.df = self.df[(self.df[date_col] >= start_date) & (self.df[date_col] <= end_date)]
            filtered_len = len(self.df)
            print(f"时间筛选: {start_date.date()} 至 {end_date.date()}")
            print(f"  原始样本数: {original_len}, 筛选后样本数: {filtered_len} (保留 {filtered_len/original_len*100:.1f}%)")
        
        self.stock_col = stock_col
        self.date_col = date_col
        self.est_win = estimation_window
        self.evt_win = event_window
        self.market_index = market_index
        
        # 路径配置
        self.base_dir = os.path.join('data', 'interim')
        self.car_dir = os.path.join(self.base_dir, 'CAR')
        self.price_path = os.path.join(self.base_dir, 'all_stock_prices.csv')
        self.market_path = os.path.join(self.base_dir, 'market_index.csv')
        
        for p in [self.base_dir, self.car_dir]:
            if not os.path.exists(p):
                os.makedirs(p)
                
        # 内存中的数据缓存
        self.market_df = None
        self.stock_df = None
        self.trade_days = None # 交易日列表 (datetime.date objects)
        
        # 初始化 Wind
        if not w.isconnected():
            w.start()

    def _format_stock_code(self, code):
        if pd.isna(code) or str(code).strip() == '':
            return None
        return str(code).strip()

    def _prepare_base_data(self):
        """
        第一步：构建所有公司代码集合，批量获取数据
        """
        print(">>> [Step 1] 准备基础数据...")
        
        # 1. 确定时间范围
        valid_dates = pd.to_datetime(self.df[self.date_col], errors='coerce').dropna()
        if len(valid_dates) == 0:
            raise ValueError("输入表格中没有有效的日期")
            
        min_date = valid_dates.min()
        # 往前多推一些天数以覆盖估计期 (假设一年250个交易日，多预留一些 buffer)
        buffer_days = abs(self.est_win[0]) + 60 
        start_date_query = (min_date - datetime.timedelta(days=int(buffer_days * 1.5))).strftime('%Y-%m-%d')
        end_date_query = datetime.datetime.today().strftime('%Y-%m-%d')
        
        print(f"    数据请求区间: {start_date_query} 至 {end_date_query}")

        # 2. 获取大盘数据 (作为交易日历基准)
        if os.path.exists(self.market_path):
            print("    加载缓存的大盘数据...")
            self.market_df = pd.read_csv(self.market_path, index_col=0, parse_dates=True)
        else:
            print(f"    正在下载大盘数据 ({self.market_index})...")
            w_market = w.wsd(self.market_index, "pct_chg", start_date_query, end_date_query, "")
            if w_market.ErrorCode != 0:
                raise Exception(f"Wind Error: {w_market.ErrorCode}")
            
            self.market_df = pd.DataFrame(w_market.Data[0], index=w_market.Times, columns=['market_ret'])
            self.market_df.index = pd.to_datetime(self.market_df.index)
            self.market_df['market_ret'] = self.market_df['market_ret'] / 100.0 # 转换为小数
            self.market_df.to_csv(self.market_path)
        
        # 生成交易日历列表 (用于快速查找)
        self.trade_days = [d.date() for d in self.market_df.index]
        self.trade_days.sort()

        # 3. 批量获取个股数据
        # 提取所有不重复的股票代码
        unique_codes = self.df[self.stock_col].apply(self._format_stock_code).dropna().unique().tolist()
        print(f"    共需获取 {len(unique_codes)} 只股票的数据")

        if os.path.exists(self.price_path):
            print("    加载缓存的个股数据矩阵 (这可能需要一点时间)...")
            self.stock_df = pd.read_csv(self.price_path, index_col=0, parse_dates=True)
            # 简单的检查，如果缓存的列数少于当前需要的，可能需要重新下载 (这里简化处理，假设缓存是最新的)
        else:
            print("    正在从 Wind 批量下载个股数据 (分批进行)...")
            # Wind wsd 一次请求太多代码可能会挂，建议分批，比如每批 100 个
            chunk_size = 100
            all_chunks = []
            
            for i in range(0, len(unique_codes), chunk_size):
                chunk_codes = unique_codes[i:i + chunk_size]
                codes_str = ",".join(chunk_codes)
                print(f"    Processing chunk {i}-{min(i+chunk_size, len(unique_codes))}...")
                
                # 注意：wsd多只股票取单个指标，返回的是 Date x Stock 的矩阵 (Data[i] 是一列)
                # "PriceAdj=F" 前复权? 一般做研究用后复权或者不复权计算收益率，Wind pct_chg 默认通常是复权后的
                # 这里显式指定复权方式，例如 "PriceAdj=F" (前复权)
                w_data = w.wsd(codes_str, "pct_chg", start_date_query, end_date_query, "PriceAdj=F")
                
                if w_data.ErrorCode != 0:
                    print(f"    Warning: Chunk {i} failed. Code: {w_data.ErrorCode}")
                    continue
                
                # 构建 DataFrame
                # Wind返回多只股票时，Data是一个列表的列表，行是字段(这里只有pct_chg)，但其实wsd多标的单指标返回结构比较特殊
                # wsd 多标的通常建议用 w.wss (截面) 或 w.wsd (序列)。
                # w.wsd 多标的单指标：Data[j] 对应第 j 只股票的时间序列
                
                chunk_df = pd.DataFrame(index=w_data.Times)
                for idx, code in enumerate(w_data.Codes):
                    chunk_df[code] = w_data.Data[idx]
                
                all_chunks.append(chunk_df)
            
            # 合并所有 chunk (按列合并)
            if all_chunks:
                self.stock_df = pd.concat(all_chunks, axis=1)
                # 转换为小数
                self.stock_df = self.stock_df / 100.0
                self.stock_df.index = pd.to_datetime(self.stock_df.index)
                # 保存
                print("    保存个股数据矩阵...")
                self.stock_df.to_csv(self.price_path)
            else:
                raise Exception("无法获取任何股票数据")
        
        print("    数据准备完成。\n")

    def _get_trade_date_loc(self, date_obj):
        """
        在交易日历中找到该日期对应的交易日索引
        规则：如果date_obj是交易日，返回该日期；否则返回之后最近的交易日
        这符合事件研究的标准做法：公告信息只有在下一个交易日才能被市场反映
        返回在 self.trade_days 中的索引
        """
        if isinstance(date_obj, pd.Timestamp):
            date_obj = date_obj.date()
            
        # 找到第一个 >= date_obj 的交易日（事件日当天或之后）
        for i, d in enumerate(self.trade_days):
            if d >= date_obj:
                return i
        return None

    def fit(self):
        """
        第二步：遍历每一行，计算回归并保存中间文件
        """
        if self.stock_df is None:
            self._prepare_base_data()
            
        print(">>> [Step 2] 开始拟合回归 (Fit)...")
        
        total = len(self.df)
        skipped = 0
        computed = 0
        
        # 注意：这里不再使用 file_fmt，因为在 fit 中已改为带 idx 的唯一文件名
        
        for idx, row in self.df.iterrows():
            if idx % 100 == 0:
                print(f"    Processing {idx}/{total}...")
                
            code = self._format_stock_code(row[self.stock_col])
            ann_date = pd.to_datetime(row[self.date_col], errors='coerce')
            
            if not code or pd.isna(ann_date):
                continue
            
            # 1. 定义文件名并检查是否存在
            # 使用事件日期区分同一股票的不同事件
            event_date_str = ann_date.strftime('%Y%m%d')
            fname_unique = f"{code}_{event_date_str}_{self.est_win[0]}_{self.est_win[1]}_{self.evt_win[0]}_{self.evt_win[1]}.csv"
            fpath = os.path.join(self.car_dir, fname_unique)
            
            if os.path.exists(fpath):
                skipped += 1
                continue
                
            # 2. 确定 T0 对应的索引
            # 逻辑：事件日定为公告日当天或之后最近的交易日（标准事件研究做法）
            t0_idx = self._get_trade_date_loc(ann_date)
            if t0_idx is None:
                # 公告日太晚，超出了下载的数据范围
                continue
                
            # 3. 计算绝对日期范围索引
            # 估计期范围
            est_start_idx = t0_idx + self.est_win[0]
            est_end_idx = t0_idx + self.est_win[1]
            # 事件期范围
            evt_start_idx = t0_idx + self.evt_win[0]
            evt_end_idx = t0_idx + self.evt_win[1]
            
            # 边界检查
            if est_start_idx < 0 or evt_end_idx >= len(self.trade_days):
                continue
                
            # 4. 提取数据
            # 获取日期列表
            est_dates = self.trade_days[est_start_idx : est_end_idx + 1]
            evt_dates = self.trade_days[evt_start_idx : evt_end_idx + 1]
            
            # 从大表提取
            # 注意：如果 code 不在 self.stock_df 列中 (比如新股或退市太久)，会报错，需处理
            if code not in self.stock_df.columns:
                continue
                
            # 提取个股和市场收益率
            # 将 date 对象转换为 pd.Timestamp 以匹配 DataFrame 索引
            try:
                # 估计期
                est_dates_ts = [pd.Timestamp(d) for d in est_dates]
                evt_dates_ts = [pd.Timestamp(d) for d in evt_dates]
                
                est_data_stock = self.stock_df.loc[est_dates_ts, code].values
                est_data_mkt = self.market_df.loc[est_dates_ts, 'market_ret'].values
                
                # 事件期
                evt_data_stock = self.stock_df.loc[evt_dates_ts, code].values
                evt_data_mkt = self.market_df.loc[evt_dates_ts, 'market_ret'].values
            except KeyError as e:
                # 日期索引对不上，可能某些日期缺失
                continue
                
            # 5. 清洗估计期数据 (去除 NaN)
            mask = ~np.isnan(est_data_stock) & ~np.isnan(est_data_mkt)
            y = est_data_stock[mask]
            x = est_data_mkt[mask]
            
            if len(y) < 15: # 估计期有效数据太少
                continue
                
            # 6. 跑回归 (CAPM)
            # y = alpha + beta * x
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # 计算估计期的残差标准误 (Residual Standard Error)
            y_pred = intercept + slope * x
            residuals = y - y_pred
            # n-2 degrees of freedom
            rmse = np.sqrt(np.sum(residuals**2) / (len(y) - 2))
            
            # 保存一些用于显著性检验的统计量
            est_n = len(y)
            est_mkt_mean = np.mean(x)
            est_mkt_var = np.sum((x - est_mkt_mean)**2)
            
            # 7. 计算完整的时间窗口的 AR（估计期 + 中间期 + 事件窗口期）
            # 确定完整范围：从估计期开始到事件窗口结束
            full_start_idx = est_start_idx
            full_end_idx = evt_end_idx
            
            # 获取完整日期列表
            full_dates = self.trade_days[full_start_idx : full_end_idx + 1]
            full_dates_ts = [pd.Timestamp(d) for d in full_dates]
            
            try:
                # 提取完整范围的数据
                full_data_stock = self.stock_df.loc[full_dates_ts, code].values
                full_data_mkt = self.market_df.loc[full_dates_ts, 'market_ret'].values
            except KeyError:
                continue
            
            # 计算完整范围的 AR
            expected_ret_full = intercept + slope * full_data_mkt
            ar_full = full_data_stock - expected_ret_full
            
            # 构建输出表格
            # 包含：日期, 相对天数 T, 实际收益, 市场收益, AR, 以及回归参数(重复存储以便读取)
            # 相对天数从估计期开始算起
            relative_days_full = range(self.est_win[0], self.evt_win[1] + 1)
            
            out_df = pd.DataFrame({
                'date': full_dates,
                'T': relative_days_full,
                'stock_ret': full_data_stock,
                'market_ret': full_data_mkt,
                'AR': ar_full,
                # 元数据
                'alpha': intercept,
                'beta': slope,
                'rmse': rmse,
                'est_n': est_n,
                'est_mkt_mean': est_mkt_mean,
                'est_mkt_var': est_mkt_var
            })
            
            out_df.to_csv(fpath, index=False)
            computed += 1
            
        print(f"Fit 完成。新计算: {computed}, 跳过(已存在): {skipped}")

    def calculate_car(self, intervals, with_significance=True):
        """
        第三步：根据指定的子区间计算 CAR
        
        参数:
        intervals: list of tuple, e.g. [(-1, 1), (-2, 2), (0, 3)]
        with_significance: bool, 是否计算 t值, p值
        """
        print(">>> [Step 3] 计算 CAR 统计量...")
        
        # 准备结果列
        result_cols = {}
        for start, end in intervals:
            suffix = f"[{start},{end}]"
            result_cols[f"CAR{suffix}"] = []
            if with_significance:
                result_cols[f"T_stat{suffix}"] = []
                result_cols[f"P_val{suffix}"] = []
                result_cols[f"Sig{suffix}"] = [] # 0/1

        results_list = [] # 暂存每一行的结果字典
        total = len(self.df)
        
        for idx, row in self.df.iterrows():
            # 添加进度提示
            if idx % 500 == 0:
                print(f"    Processing {idx}/{total}...")
            
            row_res = {}
            code = self._format_stock_code(row[self.stock_col])
            ann_date = pd.to_datetime(row[self.date_col], errors='coerce')
            
            # 默认填充 NaN
            for k in result_cols.keys():
                row_res[k] = np.nan
                
            if not code or pd.isna(ann_date):
                results_list.append(row_res)
                print(f"  Warning: 股票代码或公告日不存在，跳过计算{code}。")
                continue
                
            # 使用与 fit 中相同的唯一文件名格式（基于事件日期）
            event_date_str = ann_date.strftime('%Y%m%d')
            fname_unique = f"{code}_{event_date_str}_{self.est_win[0]}_{self.est_win[1]}_{self.evt_win[0]}_{self.evt_win[1]}.csv"
            fpath = os.path.join(self.car_dir, fname_unique)
            
            if not os.path.exists(fpath):
                results_list.append(row_res)
                print(f"   Warning: 中间文件不存在，跳过计算{code}。")
                continue
                
            # 读取中间文件
            data = pd.read_csv(fpath)
            
            # 提取元数据 (取第一行即可)
            rmse = data['rmse'].iloc[0]
            est_n = data['est_n'].iloc[0]
            est_mkt_mean = data['est_mkt_mean'].iloc[0]
            est_mkt_var = data['est_mkt_var'].iloc[0]
            
            for start_t, end_t in intervals:
                suffix = f"[{start_t},{end_t}]"
                
                # 筛选区间
                mask = (data['T'] >= start_t) & (data['T'] <= end_t)
                sub_data = data[mask]
                
                if len(sub_data) == 0:
                    continue
                
                # 计算 CAR
                # 注意处理 NaN (有些日子停牌可能导致 AR 为 NaN)
                valid_ar = sub_data['AR'].dropna()
                if len(valid_ar) == 0:
                    continue
                    
                car = valid_ar.sum()
                row_res[f"CAR{suffix}"] = car
                
                if with_significance:
                    # 计算 CAR 的方差 (标准 Patell 方法或简化方法)
                    # 这里使用标准的事件研究法方差公式：
                    # Var(CAR) = L * sigma^2 * (1 + 1/M + (Rm_window_mean - Rm_est_mean)^2 / Rm_est_var)
                    # 其中 L 是窗口长度 (非NaN的天数)
                    
                    L = len(valid_ar)
                    # 获取该窗口期内的市场收益率用于调整方差
                    mkt_rets_window = sub_data.loc[valid_ar.index, 'market_ret']
                    mkt_window_mean = mkt_rets_window.mean()
                    
                    # 修正项 (Correction Factor)
                    # 如果 est_mkt_var 为 0 (极少见)，保护一下
                    correction = 0
                    if est_mkt_var > 1e-9:
                        correction = (mkt_window_mean - est_mkt_mean)**2 / est_mkt_var
                        
                    var_car = (L * (rmse**2)) * (1 + 1/est_n + correction)
                    se_car = np.sqrt(var_car)
                    
                    if se_car > 1e-9:
                        t_stat = car / se_car
                        # 双尾检验
                        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=est_n-2))
                        is_sig = 1 if p_val < 0.05 else 0 # 5% 显著性
                    else:
                        t_stat = np.nan
                        p_val = np.nan
                        is_sig = 0
                        
                    row_res[f"T_stat{suffix}"] = t_stat
                    row_res[f"P_val{suffix}"] = p_val
                    row_res[f"Sig{suffix}"] = is_sig
            
            results_list.append(row_res)
        
        print(f"CAR 计算完成。共处理 {total} 行数据。")
        
        # 将结果合并回原 DataFrame
        res_df = pd.DataFrame(results_list, index=self.df.index)
        final_df = pd.concat([self.df, res_df], axis=1)
        
        return final_df

    def calculate_caar(self, group_col):
        """
        计算分组的AAR和CAAR (Average Abnormal Return & Cumulative Average Abnormal Return)
        
        参数:
        group_col: str, 分组列名（如"Post"），该列应为0/1变量
        
        返回:
        DataFrame: 包含各组AAR和CAAR及显著性的汇总表
                  列：T, AAR_0, AAR_0_Pval, CAAR_0, CAAR_0_Pval, 
                      AAR_1, AAR_1_Pval, CAAR_1, CAAR_1_Pval
                  行：事件窗口期内的每个交易日
        """
        print(f">>> 计算分组AAR和CAAR (按 {group_col} 列分组)...")
        
        # 检查分组列是否存在
        if group_col not in self.df.columns:
            raise ValueError(f"列 '{group_col}' 不存在于数据表中")
        
        # 获取唯一的分组值并排序
        unique_groups = sorted(self.df[group_col].dropna().unique())
        print(f"  发现分组: {unique_groups}")
        
        # 为每个分组收集所有交易日的AR数据
        # 结构: {group_val: {T: [list of AR values]}}
        group_ar_data = {group_val: {} for group_val in unique_groups}
        
        for group_val in unique_groups:
            # 筛选该组的数据
            df_group = self.df[self.df[group_col] == group_val]
            print(f"  处理 {group_col}={group_val}, 样本数: {len(df_group)}")
            
            success_count = 0
            
            for idx, row in df_group.iterrows():
                code = self._format_stock_code(row[self.stock_col])
                ann_date = pd.to_datetime(row[self.date_col], errors='coerce')
                
                if not code or pd.isna(ann_date):
                    continue
                
                # 构造文件名
                event_date_str = ann_date.strftime('%Y%m%d')
                fname = f"{code}_{event_date_str}_{self.est_win[0]}_{self.est_win[1]}_{self.evt_win[0]}_{self.evt_win[1]}.csv"
                fpath = os.path.join(self.car_dir, fname)
                
                if not os.path.exists(fpath):
                    continue
                
                try:
                    data = pd.read_csv(fpath)
                    # 只保留事件窗口期的数据
                    event_data = data[(data['T'] >= self.evt_win[0]) & (data['T'] <= self.evt_win[1])]
                    
                    # 将该股票的AR数据按T分组存储
                    for _, row_data in event_data.iterrows():
                        t = int(row_data['T'])
                        ar = row_data['AR']
                        if not np.isnan(ar):
                            if t not in group_ar_data[group_val]:
                                group_ar_data[group_val][t] = []
                            group_ar_data[group_val][t].append(ar)
                    
                    success_count += 1
                except Exception as e:
                    continue
            
            print(f"    成功读取 {success_count} 个样本的AR数据")
        
        # 计算每个交易日的AAR和显著性
        results = []
        all_t_values = sorted(set().union(*[set(d.keys()) for d in group_ar_data.values()]))
        
        print(f"  计算AAR和CAAR（共 {len(all_t_values)} 个交易日）...")
        
        for t in all_t_values:
            row_result = {'T': t}
            
            for group_val in unique_groups:
                group_label = f"{int(group_val)}"
                
                # 获取该组在T日的所有AR
                ar_list = group_ar_data[group_val].get(t, [])
                
                if len(ar_list) > 1:
                    # 计算AAR（平均AR）
                    aar = np.mean(ar_list)
                    # t检验：H0: AAR = 0
                    t_stat, p_val = stats.ttest_1samp(ar_list, 0)
                else:
                    aar = np.nan
                    p_val = np.nan
                
                # 存储AAR结果
                row_result[f'AAR_{group_label}'] = aar
                row_result[f'AAR_{group_label}_Pval'] = p_val
                row_result[f'N_{group_label}'] = len(ar_list)
            
            results.append(row_result)
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(results).sort_values('T').reset_index(drop=True)
        
        # 计算CAAR（累计AAR）及其显著性
        for group_val in unique_groups:
            group_label = f"{int(group_val)}"
            
            # 累计求和AAR得到CAAR
            result_df[f'CAAR_{group_label}'] = result_df[f'AAR_{group_label}'].cumsum()
            
            # CAAR的显著性检验：对累计到当前T的所有AR进行t检验
            caar_pvals = []
            for i in range(len(result_df)):
                t_current = result_df.loc[i, 'T']
                # 收集从事件窗口开始到当前T的所有AR
                all_ar_up_to_t = []
                for t in all_t_values[:i+1]:
                    all_ar_up_to_t.extend(group_ar_data[group_val].get(t, []))
                
                if len(all_ar_up_to_t) > 1:
                    _, p_val = stats.ttest_1samp(all_ar_up_to_t, 0)
                else:
                    p_val = np.nan
                
                caar_pvals.append(p_val)
            
            result_df[f'CAAR_{group_label}_Pval'] = caar_pvals
        
        # 重新排列列顺序
        cols_order = ['T']
        for group_val in unique_groups:
            group_label = f"{int(group_val)}"
            cols_order.extend([
                f'AAR_{group_label}',
                f'AAR_{group_label}_Pval',
                f'CAAR_{group_label}',
                f'CAAR_{group_label}_Pval',
                f'N_{group_label}'
            ])
        result_df = result_df[cols_order]
        
        print(f"\nAAR和CAAR 计算完成！")
        return result_df

# ================= 使用示例 =================

if __name__ == "__main__":
    # 假设你已经有了一个包含 Stkcd, AnnDate 和分组列的 DataFrame
    # df = pd.read_excel("your_data.xlsx")
    
    # 为了演示，我们构造一个假的
    df = pd.DataFrame({
        'Stkcd': ['000001.SZ', '600000.SH', '000002.SZ', '000004.SZ'],
        'AnnDate': ['2023-05-10', '2023-06-15', '2023-05-20', '2023-07-01'],
        'Post': [0, 1, 0, 1]  # 分组变量：0=政策前，1=政策后
    })
    
    # 1. 初始化 (设定最大的估计期和事件期范围)
    #    fit 的 event_window 要设得比 calculate_car 想要的大，
    #    比如你想算 [-1, 1], [-2, 2]，那 fit 这里设 [-5, 5] 比较稳妥
    calculator = FastBatchCARCalculator(
        df, 
        stock_col='Stkcd', 
        date_col='AnnDate',
        estimation_window=(-120, -21), # 估计期
        event_window=(-10, 10),         # 最大的事件计算范围
        date_range=('2020-01-01', '2023-12-31')  # 可选：只分析2020-2023年的事件
    )
    
    # 2. 运行拟合 (这一步会下载数据、计算回归、生成中间 csv)
    #    如果数据已经下载过，且 csv 已经生成过，这一步会飞快
    calculator.fit()
    
    # 3. 计算个股 CAR（可选）
    #    这里传入你想要计算的具体子区间
    intervals = [(-1, 1), (-2, 2), (-5, 5), (-10, 10), (0, 1), (0, 5)]
    result_df = calculator.calculate_car(intervals, with_significance=True)
    print(result_df.head())
    
    # 4. 计算分组 AAR 和 CAAR（新功能）
    #    按照 Post 列分组，计算各组在事件窗口期每个交易日的 AAR 和累计 CAAR
    aar_caar_df = calculator.calculate_caar(group_col='Post')
    
    print("\n分组AAR和CAAR结果:")
    print(aar_caar_df)
    # aar_caar_df.to_excel("aar_caar_results.xlsx", index=False)