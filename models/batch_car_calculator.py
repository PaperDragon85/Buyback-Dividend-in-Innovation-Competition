import pandas as pd
import numpy as np
import os
from WindPy import w
from scipy import stats
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class BatchCARCalculator:
    """
    批量累计超额收益率（CAR）计算器
    """
    
    def __init__(self, df, stock_col="证券代码", date_col="BPAmtDy", estimation_window=(-120, -30), event_window=(-5, 5), market_index="000001.SH"):
        """
        初始化批量 CAR 计算器
        
        参数:
        df: DataFrame, 包含股票数据
        stock_col: str, 股票代码列名
        date_col: str, 日期列名
        estimation_window: tuple, 估计窗口期，默认 (-120, -30)
        event_window: tuple, 事件窗口期，默认 (-5, 5)
        market_index: str, 市场基准指数代码，默认为上证指数 '000001.SH'
        """
        self.df = df
        self.stock_col = stock_col
        self.date_col = date_col
        self.estimation_window = estimation_window
        self.event_window = event_window
        self.market_index = market_index
        self.data_dir = os.path.join('data', 'interim', 'CAR')
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # 初始化 Wind
        if not w.isconnected():
            w.start()

    def _format_stock_code(self, code):
        """处理股票代码（仅检查缺失值，不进行格式化）"""
        if pd.isna(code) or str(code).strip() == '':
            return None
        return str(code).strip() 

    def _get_event_date(self, stock_code, announcement_date):
        """确定事件日 T=0"""
        # 检查公告日是否有股票价格数据
        try:
            price_data = w.wsd(stock_code, "close", announcement_date, announcement_date, "")
            if price_data.ErrorCode == 0 and len(price_data.Data[0]) > 0 and not pd.isna(price_data.Data[0][0]):
                return price_data.Times[0]
            else:
                # 公告日休市，获取之后第一个交易日
                next_trading = w.tdaysoffset(1, announcement_date, "")
                if next_trading.ErrorCode == 0 and len(next_trading.Data[0]) > 0:
                    return next_trading.Data[0][0]
        except Exception as e:
            print(f"获取事件日失败 {stock_code} {announcement_date}: {e}")
        return None

    def _fetch_market_data(self, min_date, max_date, refresh_data=False):
        """获取市场指数数据"""
        market_file = os.path.join(self.data_dir, 'market_returns.csv')
        if not refresh_data and os.path.exists(market_file):
            print("使用缓存的市场指数数据")
            return

        print("正在获取市场指数数据...")
        # 扩大范围以确保覆盖
        start_date = w.tdaysoffset(-250, min_date, "").Data[0][0] # 约1年前
        end_date = w.tdaysoffset(30, max_date, "").Data[0][0] # 往后多取一点
        
        market_data = w.wsd(self.market_index, "pct_chg", start_date, end_date, "")
        if market_data.ErrorCode != 0:
            raise ValueError(f"无法获取市场数据: {market_data.ErrorCode}")
            
        df_market = pd.DataFrame({
            'date': [d.date() if hasattr(d, 'date') else d for d in market_data.Times],
            'market_return': np.array(market_data.Data[0]) / 100
        })
        # 去重并保存
        df_market = df_market.drop_duplicates(subset=['date'])
        df_market.to_csv(os.path.join(self.data_dir, 'market_returns.csv'), index=False)
        return df_market

    def _fetch_row_data(self, index, row, refresh_data=False):
        """处理单行数据获取"""
        save_path = os.path.join(self.data_dir, f'row_{index}.csv')
        if not refresh_data and os.path.exists(save_path):
            return True

        stock_code = self._format_stock_code(row[self.stock_col])
        ann_date = row[self.date_col]
        
        if not stock_code or pd.isna(ann_date):
            return None

        # 格式化日期
        if isinstance(ann_date, pd.Timestamp):
            ann_date_str = ann_date.strftime('%Y-%m-%d')
        else:
            ann_date_str = str(ann_date).strip()

        # 1. 确定事件日 T=0
        event_date = self._get_event_date(stock_code, ann_date_str)
        if not event_date:
            return None

        # 2. 计算需要的交易日范围
        start_t = self.estimation_window[0]
        end_t = self.event_window[1]
        
        # 获取范围内的交易日
        # 为了保险，前后多取几天
        extended_start = w.tdaysoffset(start_t - 10, event_date, "")
        extended_end = w.tdaysoffset(end_t + 10, event_date, "")
        
        if extended_start.ErrorCode != 0 or extended_end.ErrorCode != 0:
            return None
            
        s_date = extended_start.Data[0][0]
        e_date = extended_end.Data[0][0]
        
        all_trade_days = w.tdays(s_date, e_date, "")
        if all_trade_days.ErrorCode != 0:
            return None
            
        trade_day_list = all_trade_days.Data[0]
        
        # 找到事件日索引
        event_date_normalized = event_date.date() if hasattr(event_date, 'date') else event_date
        normalized_trade_days = [d.date() if hasattr(d, 'date') else d for d in trade_day_list]
        
        try:
            event_index = normalized_trade_days.index(event_date_normalized)
        except ValueError:
            return None # 事件日不在交易日列表中
            
        # 3. 获取股票收益率
        stock_data = w.wsd(stock_code, "pct_chg", s_date, e_date, "")
        if stock_data.ErrorCode != 0:
            return None
            
        stock_returns = np.array(stock_data.Data[0]) / 100
        stock_dates = [d.date() if hasattr(d, 'date') else d for d in stock_data.Times]
        
        # 构建数据
        data_list = []
        # 确保日期对齐
        stock_dict = dict(zip(stock_dates, stock_returns))
        
        for i, date in enumerate(normalized_trade_days):
            t = i - event_index
            
            # 只保留感兴趣的区间
            if t < self.estimation_window[0] or t > self.event_window[1]:
                continue
                
            in_estimation = self.estimation_window[0] <= t <= self.estimation_window[1]
            in_event = self.event_window[0] <= t <= self.event_window[1]
            
            ret = stock_dict.get(date, np.nan)
            
            data_list.append({
                'date': date,
                'stock_return': ret,
                'T': t,
                'in_estimation': in_estimation,
                'in_event': in_event
            })
            
        df_row = pd.DataFrame(data_list)
        save_path = os.path.join(self.data_dir, f'row_{index}.csv')
        df_row.to_csv(save_path, index=False)
        return True

    def _calculate_single_row(self, index, market_df):
        """计算单行的 CAR"""
        file_path = os.path.join(self.data_dir, f'row_{index}.csv')
        if not os.path.exists(file_path):
            return index, np.nan, np.nan
            
        try:
            df_stock = pd.read_csv(file_path)
            # 转换日期格式以匹配
            df_stock['date'] = pd.to_datetime(df_stock['date']).dt.date
            
            # 合并市场数据
            # market_df 的 date 应该是 date 对象
            df_merged = pd.merge(df_stock, market_df, on='date', how='inner')
            
            # 估计期数据
            est_data = df_merged[df_merged['in_estimation'] == True]
            if len(est_data) < 10: # 数据太少
                return index, np.nan, np.nan
                
            # 去除 NaN
            est_data = est_data.dropna(subset=['stock_return', 'market_return'])
            if len(est_data) < 10:
                return index, np.nan, np.nan
                
            # CAPM 回归
            X = est_data['market_return'].values
            y = est_data['stock_return'].values
            n = len(y)
            
            # 添加常数项
            X_mat = np.column_stack([np.ones(n), X])
            
            # OLS
            params = np.linalg.lstsq(X_mat, y, rcond=None)[0]
            alpha, beta = params[0], params[1]
            
            # 残差标准误
            y_pred = X_mat @ params
            residuals = y - y_pred
            residual_std = np.std(residuals, ddof=2)
            
            # 事件期数据
            evt_data = df_merged[df_merged['in_event'] == True]
            if len(evt_data) == 0:
                return index, np.nan, np.nan
                
            evt_data = evt_data.dropna(subset=['stock_return', 'market_return'])
            
            # 计算超额收益
            evt_market = evt_data['market_return'].values
            evt_stock = evt_data['stock_return'].values
            
            expected_returns = alpha + beta * evt_market
            abnormal_returns = evt_stock - expected_returns
            
            # 计算 CAR
            car = np.sum(abnormal_returns)
            
            # 计算显著性
            # CAR 的标准误
            X_mean = np.mean(X)
            X_var = np.sum((X - X_mean) ** 2)
            
            var_ar_sum = 0
            for rm in evt_market:
                var_ar = (residual_std ** 2) * (1 + 1/n + (rm - X_mean)**2 / X_var)
                var_ar_sum += var_ar
            
            se_car = np.sqrt(var_ar_sum)
            
            t_stat = car / se_car
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            
            is_significant = 1 if p_value < 0.05 else 0 # 默认 95%
            
            return index, car, is_significant
            
        except Exception as e:
            # print(f"计算错误 index {index}: {e}")
            return index, np.nan, np.nan

    def run(self, add_car=True, add_significance=True, refresh_data=False):
        """
        执行批量计算
        
        参数:
        add_car: bool, 是否添加 CAR 列
        add_significance: bool, 是否添加显著性列
        refresh_data: bool, 是否强制重新获取数据，默认为 False
        """
        # 1. 获取市场数据
        print("Step 1: 获取市场数据...")
        # 过滤掉日期为空的行来计算日期范围
        valid_dates = pd.to_datetime(self.df[self.date_col], errors='coerce').dropna()
        if len(valid_dates) == 0:
            raise ValueError("没有有效的日期数据")
            
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        self._fetch_market_data(min_date, max_date, refresh_data)
        
        # 2. 获取个股数据
        print("Step 2: 获取个股数据 (Wind)...")
        
        total_rows = len(self.df)
        for index, row in self.df.iterrows():
            if index % 100 == 0:
                print(f"Processing row {index}/{total_rows}")
            self._fetch_row_data(index, row, refresh_data)
            
        # 3. 多线程计算
        print("Step 3: 多线程计算 CAR...")
        market_df = pd.read_csv(os.path.join(self.data_dir, 'market_returns.csv'))
        market_df['date'] = pd.to_datetime(market_df['date']).dt.date
        
        results = {}
        max_workers = os.cpu_count() or 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._calculate_single_row, idx, market_df): idx for idx in self.df.index}
            
            for future in as_completed(futures):
                idx, car, sig = future.result()
                results[idx] = (car, sig)
                
        # 4. 整合结果
        print("Step 4: 整合结果...")
        car_series = pd.Series(index=self.df.index, dtype=float)
        sig_series = pd.Series(index=self.df.index, dtype=float)
        
        for idx, (car, sig) in results.items():
            car_series[idx] = car
            sig_series[idx] = sig
            
        result_df = self.df.copy()
        if add_car:
            result_df['CAR'] = car_series
        if add_significance:
            result_df['CAR_Significant'] = sig_series
            
        return result_df

if __name__ == "__main__":
    # 测试代码
    # 构造一个简单的测试 DataFrame
    test_df = pd.DataFrame({
        'Stkcd': ['000001.SZ', '600000.SH', None, '000002.SZ'],
        'AnnDate': ['2024-03-15', '2024-03-20', '2024-03-21', None]
    })
    
    # 初始化时传入 df 和列名
    calculator = BatchCARCalculator(test_df, 'Stkcd', 'AnnDate')
    res = calculator.run(add_car=True, add_significance=True)
    print(res)
