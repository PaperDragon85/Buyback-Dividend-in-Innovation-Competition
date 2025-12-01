import akshare as ak
import pandas as pd
import statsmodels.formula.api as smf
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
import os
from scipy.stats import ttest_1samp
if not os.path.exists('./CAR_res'):
    os.mkdir('./CAR_res')
if not os.path.exists('./plot'):
    os.mkdir('./plot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def ols_return(stock, date_time, rt_code=None):
    start_time = date_time[0].strftime('%Y%m%d')
    end_time = date_time[-1].strftime('%Y%m%d')
    if rt_code == None:
        if stock.startswith(('43', '60')):
            rt_code = 'sz399001'
        elif stock.startswith('00'):
            rt_code = 'sh000001'
        elif stock.startswith(('30', '83', '92')):
            rt_code = 'sz399006'
        else:
            rt_code = 'sh688050'
    df_stock = ak.stock_zh_a_hist(symbol=stock, adjust="qfq", start_date=start_time, end_date=end_time)
    df_rm = ak.stock_zh_index_daily(symbol=rt_code)
    df_stock['Ri'] = df_stock['收盘'].pct_change()
    df_rm['Rm'] = df_rm['close'].pct_change()
    df_rm = df_rm.rename(columns={'date': '日期'})
    df = pd.merge(df_stock[['日期', 'Ri']], df_rm[['日期', 'Rm']], on='日期').dropna()
    mod = smf.ols(formula='Ri~Rm', data=df)
    res = mod.fit()
    return res.params

def ols_prediction(stock, window_time, alpha, beta, rt_code=None):
    start_time = (window_time[0] - pd.Timedelta(days=1)).strftime('%Y%m%d')
    end_time = window_time[-1].strftime('%Y%m%d')
    if rt_code == None:
        if stock.startswith(('43', '60')):
            rt_code = 'sz399001'
        elif stock.startswith('00'):
            rt_code = 'sh000001'
        elif stock.startswith(('30', '83', '92')):
            rt_code = 'sz399006'
        else:
            rt_code = 'sh688050'
    df_stock = ak.stock_zh_a_hist(symbol=stock, adjust="qfq", start_date=start_time, end_date=end_time)
    df_rm = ak.stock_zh_index_daily(rt_code)
    df_stock['Ri'] = df_stock['收盘'].pct_change()
    df_rm['Rm'] = df_rm['close'].pct_change()
    df_rm = df_rm.rename(columns={'date': '日期'})
    df = pd.merge(df_stock[['日期', 'Ri']], df_rm[['日期', 'Rm']], on='日期').dropna()
    df['pRi'] = df['Rm'] * beta + alpha
    df['alpha'] = alpha
    df['beta'] = beta
    df['AR'] = df['Ri'] - df['pRi']
    df['CAR'] = df['AR'].cumsum()
    return df

def significance(series, t=False, p=False):
    series = list(series)
    t_stat, p_value = ttest_1samp(series, popmean=0)
    if t and p:
        return t_stat, p_value
    if t:
        return t_stat
    if p:
        return p_value
    else:
        return t_stat, p_value
def validation():
    stock = entry_stock.get()
    date = entry_date.get()
    if date in [dt.datetime.strftime(i, '%Y-%m-%d') for i in ak.stock_zh_a_hist(symbol=stock, adjust="qfq")['日期']]:
        entry_date.delete(0, tk.END)
        byd = Stock(stock, date)
        byd.ols(excel=True, res=True, plot_AR=True, plot_CAR=True)
    else:
        messagebox.showinfo("错误", "这天股票不开市")
        entry_date.delete(0, tk.END)
class Stock:
    def __init__(self, stock, handle_time, estimate_time=None, window_time=None):
        self.stock = stock
        self.handle_time = handle_time
        if not estimate_time:
            estimate_time = 100
        if not window_time:
            window_time = 10
        time_series = pd.Series([dt.datetime.strftime(i, '%Y-%m-%d') for i in ak.stock_zh_a_hist(symbol=stock, adjust="qfq")['日期']])
        while True:
            try:
                location = time_series[time_series == handle_time].index[0]
            except KeyError:
                handle_time = dt.datetime.strptime(handle_time, '%Y-%m-%d') + dt.timedelta(days=1)
                handle_time = dt.datetime.strftime(handle_time, '%Y-%m-%d')
            else:
                break
        self.window_time = pd.date_range(time_series.iloc[location - window_time], time_series.iloc[location + window_time])
        self.estimate_time = pd.date_range(time_series.iloc[location - window_time - estimate_time], time_series.iloc[location - window_time - 1])
    def ols(self, rt_code=None, excel=False, res=False, plot_AR=False, plot_CAR=False):
        df_res = ols_return(self.stock, self.estimate_time, rt_code)
        alpha = df_res.iloc[0]
        beta = df_res.iloc[1]
        df_CAR = ols_prediction(self.stock, self.window_time, alpha, beta, rt_code)
        df_CAR['AR_t'] = significance(df_CAR['AR'], t=True)
        df_CAR['AR_p'] = significance(df_CAR['AR'], p=True)
        df_CAR['CAR_t'] = significance(df_CAR['CAR'], t=True)
        df_CAR['CAR_p'] = significance(df_CAR['CAR'], p=True)
        if excel == True:
            df_CAR.to_excel(f'./CAR_res/{self.stock}_{self.handle_time}.xlsx')
            print(f'./CAR_res/{self.stock}_{self.handle_time}.xlsx已生成')
        if res == True:
            print(df_res)
        if plot_AR == True:
            sns.lineplot(data=df_CAR, x='日期', y='AR', label='AR')
        if plot_CAR == True:
            sns.lineplot(data=df_CAR, x='日期', y='CAR', label='CAR')
        if plot_AR or plot_CAR:
            plt.xticks(rotation=45)
            plt.savefig(f'./plot/{self.stock}_{self.handle_time}.png')
            print(f'./plot/{self.stock}_{self.handle_time}.png已生成')
        return df_CAR
# byd = Stock('sz002594', '2012-01-09', 100, 10)
# byd.ols(excel=True, res=True, plot_AR=True, plot_CAR=True)
if __name__ == '__main__':
    root = tk.Tk()
    root.title("CAR计算器")
    label_stock = tk.Label(root, text='请输入股票代码,如002594')
    label_stock.grid(row=0, column=0, padx=10, pady=10)
    entry_stock = tk.Entry(root)
    entry_stock.grid(row=1, column=0, padx=10, pady=10)

    label_date = tk.Label(root, text='请输入日期,如2024-03-08')
    label_date.grid(row=2, column=0, padx=10, pady=10)
    entry_date = tk.Entry(root)
    entry_date.grid(row=3, column=0, padx=10, pady=10)

    login_button = tk.Button(root, text="计算", command=validation)
    login_button.grid(row=4, column=0, columnspan=2, pady=10)
    root.mainloop()