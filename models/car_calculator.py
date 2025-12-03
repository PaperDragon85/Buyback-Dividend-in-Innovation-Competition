"""
CAR (Cumulative Abnormal Return) Calculator
使用 CAPM 模型计算股票事件研究中的累计超额收益率
"""

import pandas as pd
import numpy as np
from WindPy import w
from scipy import stats
import warnings


class CARCalculator:
    """
    累计超额收益率（CAR）计算器
    
    使用 CAPM 模型进行事件研究分析，计算股票在特定事件窗口期的超额收益
    """
    
    def __init__(self, stock_code, announcement_date, estimation_window=(-120, -30), 
                 event_window=(-5, 5), market_index="000001.SH"):
        """
        初始化 CAR 计算器
        
        参数:
        stock_code: str, 股票代码，如 '000001.SZ'
        announcement_date: str, 公告日期，格式 'YYYY-MM-DD'
        estimation_window: tuple, 估计窗口期，默认 (-120, -30)，相对于事件日的交易日
        event_window: tuple, 事件窗口期，默认 (-5, 5)，相对于事件日的交易日
        market_index: str, 市场基准指数代码，默认为上证指数 '000001.SH'
        """
        # 初始化 Wind
        if not w.isconnected():
            w.start()
        
        # 检查并处理股票代码
        if pd.isna(stock_code) or stock_code is None or str(stock_code).strip() == '':
            raise ValueError(f"股票代码不能为空: {stock_code}")
        
        self.stock_code = self._format_stock_code(str(stock_code).strip())
        
        # 检查并处理日期
        if pd.isna(announcement_date) or announcement_date is None:
            raise ValueError(f"公告日期不能为空: {announcement_date}")
        
        # 统一转换日期格式为字符串
        if isinstance(announcement_date, pd.Timestamp):
            self.announcement_date = announcement_date.strftime('%Y-%m-%d')
        elif hasattr(announcement_date, 'strftime'):  # datetime 对象
            self.announcement_date = announcement_date.strftime('%Y-%m-%d')
        else:
            self.announcement_date = str(announcement_date).strip()
        
        # 验证日期格式
        if len(self.announcement_date) != 10 or self.announcement_date[4] != '-' or self.announcement_date[7] != '-':
            raise ValueError(f"日期格式不正确，应为 YYYY-MM-DD: {self.announcement_date}")
        
        self.market_index = market_index
        self.estimation_window = estimation_window
        self.event_window = event_window
        
        # 回归状态
        self.is_fitted = False
        
        # CAPM 参数
        self.alpha = None
        self.beta = None
        self.alpha_std = None
        self.beta_std = None
        self.residual_std = None
        
        # 数据存储
        self.est_stock_returns = None
        self.est_market_returns = None
        
        self.evt_stock_returns = None
        self.evt_market_returns = None
        
        # 确定 T=0 的事件日（如果公告日休市，则选择之后第一个开市日）
        self.event_date = self._get_event_date()
        
        # 构建 T 与日期的对应关系
        self.date_mapping = self._build_date_mapping()
    
    def _format_stock_code(self, code):
        """
        格式化股票代码，自动添加交易所后缀
        """
        code = code.strip().upper()
        
        # 如果已经有后缀，直接返回
        if '.' in code:
            return code
        
        # 根据代码开头判断交易所
        if code.startswith(('000', '001', '002', '003', '300', '301')):
            return f"{code}.SZ"  # 深圳
        elif code.startswith(('600', '601', '603', '605', '688', '689')):
            return f"{code}.SH"  # 上海
        else:
            raise ValueError(f"无法识别股票代码: {code}，请手动指定交易所后缀（.SZ 或 .SH）")
    
    def _get_event_date(self):
        """
        确定事件日 T=0：
        - 如果公告日是交易日（有价格），则 T=0 为公告日
        - 如果公告日休市（无价格），则 T=0 为公告日之后第一个交易日
        """
        # 检查公告日是否有股票价格数据
        price_data = w.wsd(self.stock_code, "close", self.announcement_date, self.announcement_date, "")
        
        # 如果有价格数据且不为空，说明是交易日
        if price_data.ErrorCode == 0 and len(price_data.Data[0]) > 0 and not pd.isna(price_data.Data[0][0]):
            # 公告日是交易日
            event_date = price_data.Times[0]
        else:
            # 公告日休市，获取之后第一个交易日
            next_trading = w.tdaysoffset(1, self.announcement_date, "")
            if next_trading.ErrorCode != 0 or len(next_trading.Data[0]) == 0:
                raise ValueError(f"无法获取公告日之后的交易日: {self.announcement_date}")
            event_date = next_trading.Data[0][0]
        
        return event_date
    
    def _build_date_mapping(self):
        """
        构建 T 与真实日期的对应关系
        
        返回:
        pd.DataFrame: 包含以下列的数据框
            - 'T': int, 相对于事件日的交易日序号
            - 'date': datetime, 真实日期
            - 'in_estimation': bool, 是否在估计期内
            - 'in_event': bool, 是否在事件窗口期内
        """
        # 计算需要的交易日范围
        start_t = self.estimation_window[0]
        end_t = self.event_window[1]
        
        # 获取足够的交易日（多取一些以防万一）
        extended_start = w.tdaysoffset(start_t - 10, self.event_date, "")
        extended_end = w.tdaysoffset(end_t + 10, self.event_date, "")
        
        all_trade_days = w.tdays(extended_start.Data[0][0], extended_end.Data[0][0], "")
        if all_trade_days.ErrorCode != 0:
            raise ValueError(f"无法获取交易日序列")
        
        trade_day_list = all_trade_days.Data[0]
        
        # 统一日期格式进行比较（转换为 datetime.date）
        event_date_normalized = self.event_date
        if hasattr(event_date_normalized, 'date'):
            event_date_normalized = event_date_normalized.date()
        
        # 标准化交易日列表中的日期
        normalized_trade_days = []
        for d in trade_day_list:
            if hasattr(d, 'date'):
                normalized_trade_days.append(d.date())
            else:
                normalized_trade_days.append(d)
        
        # 找到事件日在交易日序列中的位置
        try:
            event_index = normalized_trade_days.index(event_date_normalized)
        except ValueError:
            # 如果找不到，尝试找最接近的日期
            raise ValueError(f"事件日 {self.event_date} 不在交易日列表中。交易日列表范围: {trade_day_list[0]} 到 {trade_day_list[-1]}")
        
        # 构建 T 与日期的映射
        mapping_data = []
        for i, date in enumerate(trade_day_list):
            t = i - event_index
            in_estimation = self.estimation_window[0] <= t <= self.estimation_window[1]
            in_event = self.event_window[0] <= t <= self.event_window[1]
            
            mapping_data.append({
                'T': t,
                'date': date,
                'in_estimation': in_estimation,
                'in_event': in_event
            })
        
        df_mapping = pd.DataFrame(mapping_data)
        return df_mapping
    
    def _load_data(self):
        """
        加载估计窗口和事件窗口的数据
        """
        # 从 date_mapping 中筛选估计期和事件期的日期
        est_dates = self.date_mapping[self.date_mapping['in_estimation']]['date'].tolist()
        evt_dates = self.date_mapping[self.date_mapping['in_event']]['date'].tolist()
        
        if len(est_dates) == 0:
            raise ValueError("估计窗口没有交易日数据")
        if len(evt_dates) == 0:
            raise ValueError("事件窗口没有交易日数据")
        
        est_start = est_dates[0]
        est_end = est_dates[-1]
        evt_start = evt_dates[0]
        evt_end = evt_dates[-1]
        
        # 获取估计窗口的数据
        est_stock_data = w.wsd(self.stock_code, "pct_chg", est_start, est_end, "")
        est_market_data = w.wsd(self.market_index, "pct_chg", est_start, est_end, "")
        
        if est_stock_data.ErrorCode != 0 or est_market_data.ErrorCode != 0:
            raise ValueError("无法获取估计窗口数据")
        
        # 获取事件窗口的数据
        evt_stock_data = w.wsd(self.stock_code, "pct_chg", evt_start, evt_end, "")
        evt_market_data = w.wsd(self.market_index, "pct_chg", evt_start, evt_end, "")
        
        if evt_stock_data.ErrorCode != 0 or evt_market_data.ErrorCode != 0:
            raise ValueError("无法获取事件窗口数据")
        
        # 保存数据到属性
        self.est_stock_returns = np.array(est_stock_data.Data[0]) / 100
        self.est_market_returns = np.array(est_market_data.Data[0]) / 100
        
        self.evt_stock_returns = np.array(evt_stock_data.Data[0]) / 100
        self.evt_market_returns = np.array(evt_market_data.Data[0]) / 100
        
        # 移除估计窗口中的 NaN 值
        valid_idx = ~(np.isnan(self.est_stock_returns) | np.isnan(self.est_market_returns))
        self.est_stock_returns = self.est_stock_returns[valid_idx]
        self.est_market_returns = self.est_market_returns[valid_idx]
    
    def fit(self):
        """
        使用 CAPM 模型进行回归分析
        
        返回:
        dict: 包含回归结果的字典
        """
        # 加载数据
        self._load_data()
        
        # 使用 CAPM 模型估计 alpha 和 beta
        # R_i = alpha + beta * R_m + epsilon
        n = len(self.est_market_returns)
        X = np.column_stack([np.ones(n), self.est_market_returns])
        y = self.est_stock_returns
        
        # OLS 回归
        params = np.linalg.lstsq(X, y, rcond=None)[0]
        self.alpha, self.beta = params[0], params[1]
        
        # 计算残差和标准误
        y_pred = X @ params
        residuals = y - y_pred
        self.residual_std = np.std(residuals, ddof=2)
        
        # 计算参数的标准误
        XtX_inv = np.linalg.inv(X.T @ X)
        var_params = self.residual_std ** 2 * np.diag(XtX_inv)
        self.alpha_std = np.sqrt(var_params[0])
        self.beta_std = np.sqrt(var_params[1])
        
        # 计算 R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # 计算 t 统计量和 p 值
        t_alpha = self.alpha / self.alpha_std
        t_beta = self.beta / self.beta_std
        p_alpha = 2 * (1 - stats.t.cdf(abs(t_alpha), n - 2))
        p_beta = 2 * (1 - stats.t.cdf(abs(t_beta), n - 2))
        
        # 标记为已拟合
        self.is_fitted = True
        
        # 返回回归结果
        results = {
            'alpha': self.alpha,
            'beta': self.beta,
            'alpha_std': self.alpha_std,
            'beta_std': self.beta_std,
            'residual_std': self.residual_std,
            't_alpha': t_alpha,
            't_beta': t_beta,
            'p_alpha': p_alpha,
            'p_beta': p_beta,
            'r_squared': r_squared,
            'n_obs': n
        }
        
        return results
    
    def _ensure_fitted(self):
        """
        确保模型已经拟合，如果没有则自动拟合
        """
        if not self.is_fitted:
            warnings.warn("模型尚未拟合，自动使用默认参数进行拟合")
            self.fit()
    
    def predict_r(self, confidence_level=0.95):
        """
        预测窗口期的收益率向量及其置信区间
        
        参数:
        confidence_level: float, 置信水平，默认 0.95 (95% 置信区间)
        
        返回:
        pd.DataFrame: 包含以下列的数据框
            - 'T': int, 相对于事件日的交易日序号
            - '日期': datetime, 交易日日期
            - '实际收益率': float, 股票实际收益率（小数形式，如 0.05 表示 5%）
            - '预测收益率': float, CAPM 模型预测的收益率
            - '标准误': float, 预测的标准误
            - '置信下限_95%': float, 置信区间下限（默认 95%）
            - '置信上限_95%': float, 置信区间上限（默认 95%）
        """
        self._ensure_fitted()
        
        # 计算预测收益率
        predicted_returns = self.alpha + self.beta * self.evt_market_returns
        
        # 计算预测标准误
        n = len(self.est_market_returns)
        X_mean = np.mean(self.est_market_returns)
        X_var = np.sum((self.est_market_returns - X_mean) ** 2)
        
        # 对每个事件窗口的点计算标准误
        se_pred = []
        for rm in self.evt_market_returns:
            se = self.residual_std * np.sqrt(1 + 1/n + (rm - X_mean)**2 / X_var)
            se_pred.append(se)
        se_pred = np.array(se_pred)
        
        # 计算置信区间
        t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 2)
        ci_lower = predicted_returns - t_critical * se_pred
        ci_upper = predicted_returns + t_critical * se_pred
        
        # 获取事件窗口的日期和 T
        evt_mapping = self.date_mapping[self.date_mapping['in_event']].copy()
        
        # 构建结果数据框
        results_df = pd.DataFrame({
            'T': evt_mapping['T'].values,
            '日期': evt_mapping['date'].values,
            '实际收益率': self.evt_stock_returns,
            '预测收益率': predicted_returns,
            '标准误': se_pred,
            f'置信下限_{int(confidence_level*100)}%': ci_lower,
            f'置信上限_{int(confidence_level*100)}%': ci_upper
        })
        
        return results_df
    
    def predict_ar(self, confidence_level=0.95):
        """
        预测窗口期每天的超额收益及其置信区间
        
        参数:
        confidence_level: float, 置信水平，默认 0.95 (95% 置信区间)
        
        返回:
        pd.DataFrame: 包含以下列的数据框
            - 'T': int, 相对于事件日的交易日序号
            - '日期': datetime, 交易日日期
            - '超额收益率': float, 超额收益率 AR = 实际收益率 - 预测收益率（小数形式）
            - '标准误': float, 超额收益率的标准误
            - 't统计量': float, 超额收益率的 t 统计量
            - 'p值': float, 双侧检验的 p 值
            - '置信下限_95%': float, 置信区间下限（默认 95%）
            - '置信上限_95%': float, 置信区间上限（默认 95%）
            - '是否显著': bool, 是否在指定置信水平下显著（True/False）
        """
        self._ensure_fitted()
        
        # 计算超额收益
        expected_returns = self.alpha + self.beta * self.evt_market_returns
        abnormal_returns = self.evt_stock_returns - expected_returns
        
        # 计算标准误
        n = len(self.est_market_returns)
        X_mean = np.mean(self.est_market_returns)
        X_var = np.sum((self.est_market_returns - X_mean) ** 2)
        
        se_ar = []
        for rm in self.evt_market_returns:
            se = self.residual_std * np.sqrt(1 + 1/n + (rm - X_mean)**2 / X_var)
            se_ar.append(se)
        se_ar = np.array(se_ar)
        
        # 计算置信区间
        t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 2)
        ci_lower = abnormal_returns - t_critical * se_ar
        ci_upper = abnormal_returns + t_critical * se_ar
        
        # 计算 t 统计量和 p 值
        t_stats = abnormal_returns / se_ar
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - 2))
        
        # 获取事件窗口的日期和 T
        evt_mapping = self.date_mapping[self.date_mapping['in_event']].copy()
        
        # 构建结果数据框
        results_df = pd.DataFrame({
            'T': evt_mapping['T'].values,
            '日期': evt_mapping['date'].values,
            '超额收益率': abnormal_returns,
            '标准误': se_ar,
            't统计量': t_stats,
            'p值': p_values,
            f'置信下限_{int(confidence_level*100)}%': ci_lower,
            f'置信上限_{int(confidence_level*100)}%': ci_upper,
            '是否显著': p_values < (1 - confidence_level)
        })
        
        return results_df
    
    def predict_car(self, confidence_level=0.95):
        """
        计算累计超额收益（CAR）及其置信区间
        
        参数:
        confidence_level: float, 置信水平，默认 0.95 (95% 置信区间)
        
        返回:
        dict: 包含以下键值对的字典
            - 'CAR': float, 累计超额收益率（小数形式）
            - 'CAR_percent': float, 累计超额收益率（百分比形式）
            - 'std_error': float, CAR 的标准误
            - 't_statistic': float, t 统计量
            - 'p_value': float, 双侧检验的 p 值
            - 'confidence_level': float, 置信水平
            - 'ci_lower_95': float, 置信区间下限（键名根据置信水平变化）
            - 'ci_upper_95': float, 置信区间上限（键名根据置信水平变化）
            - 'is_significant': bool, 是否显著
            - 'event_window': tuple, 事件窗口范围
            - 'n_days': int, 事件窗口包含的交易日数
        """
        self._ensure_fitted()
        
        # 计算超额收益
        expected_returns = self.alpha + self.beta * self.evt_market_returns
        abnormal_returns = self.evt_stock_returns - expected_returns
        
        # 计算 CAR
        car = np.sum(abnormal_returns)
        
        # 计算 CAR 的标准误
        # CAR 的方差 = T * sigma^2 (假设超额收益率独立同分布)
        T = len(abnormal_returns)
        n = len(self.est_market_returns)
        
        # 更精确的方法：考虑每个 AR 的方差
        X_mean = np.mean(self.est_market_returns)
        X_var = np.sum((self.est_market_returns - X_mean) ** 2)
        
        var_ar_sum = 0
        for rm in self.evt_market_returns:
            var_ar = (self.residual_std ** 2) * (1 + 1/n + (rm - X_mean)**2 / X_var)
            var_ar_sum += var_ar
        
        se_car = np.sqrt(var_ar_sum)
        
        # 计算置信区间
        t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 2)
        ci_lower = car - t_critical * se_car
        ci_upper = car + t_critical * se_car
        
        # 计算 t 统计量和 p 值
        t_stat = car / se_car
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        # 判断显著性
        is_significant = p_value < (1 - confidence_level)
        
        results = {
            'CAR': car,
            'CAR_percent': car * 100,
            'std_error': se_car,
            't_statistic': t_stat,
            'p_value': p_value,
            'confidence_level': confidence_level,
            f'ci_lower_{int(confidence_level*100)}': ci_lower,
            f'ci_upper_{int(confidence_level*100)}': ci_upper,
            'is_significant': is_significant,
            'event_window': self.event_window,
            'n_days': T
        }
        
        return results
    
    def summary(self):
        """
        打印完整的分析摘要
        """
        print("\n" + "="*60)
        print("CAR 分析摘要")
        print("="*60)
        print(f"股票代码:     {self.stock_code}")
        print(f"公告日期:     {self.announcement_date}")
        print(f"事件日期:     {self.event_date}")
        print(f"市场基准:     {self.market_index}")
        print(f"模型状态:     {'已拟合' if self.is_fitted else '未拟合'}")
        
        if self.is_fitted:
            print(f"估计窗口:     {self.estimation_window}")
            print(f"事件窗口:     {self.event_window}")
            print(f"Alpha:        {self.alpha:.6f}")
            print(f"Beta:         {self.beta:.6f}")
        
        print("="*60 + "\n")


# 示例使用
if __name__ == "__main__":
    # 创建 CAR 计算器实例（窗口期参数在初始化时指定）
    calc = CARCalculator(
        stock_code="000001.SZ",
        announcement_date="2024-03-15",
        estimation_window=(-120, -30),
        event_window=(-5, 5)
    )
    
    # 查看日期映射
    print("\n日期映射 (前10行):")
    print(calc.date_mapping.head(10))
    
    print("\n事件日 (T=0):")
    print(calc.date_mapping[calc.date_mapping['T'] == 0])
    
    # 拟合模型（无需传参）
    fit_results = calc.fit()
    
    # 预测收益率
    r_pred = calc.predict_r()
    print("\n预测收益率:")
    print(r_pred)
    
    # 计算超额收益
    ar_results = calc.predict_ar()
    print("\n超额收益率:")
    print(ar_results)
    
    # 计算 CAR
    car_results = calc.predict_car()
    
    # 打印摘要
    calc.summary()
