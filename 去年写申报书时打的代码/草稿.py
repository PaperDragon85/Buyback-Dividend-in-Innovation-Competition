import pandas as pd
import akshare as ak
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# a = (pd.to_datetime('2023-11-12'))
# print(pd.date_range(a, periods=10))
# print(a.strftime('%Y%m%d'))
# ak.stock_zh_a_spot().to_excel('info/stock.xlsx')
# ak.stock_zh_a_spot_em().to_excel('info/stock_em.xlsx')
# ak.stock_zh_index_daily(symbol='')
# df = pd.Series([1, 1, 1, 0, 1])
# print(df.where(df == 0).stack().index)
# print(dt.datetime.strptime('2023-5-5', '%Y-%m-%d'))
# print(dt.datetime.now().strftime('%Y-%m-%d'))
# from CAR import Stock
# byd = Stock('sz002594', '2012-01-09', 100, 10)
# byd.ols(excel=True, res=True, plot_AR=True, plot_CAR=True)
# em_df = pd.read_excel('info/stock_em.xlsx', dtype={'代码': str})
# def re_start(word):
#     return word[0:2]
# code = em_df['代码'].map(re_start).unique()
# print(code)
# ak.stock_zh_index_daily(symbol='sh688050').to_excel('index_bz.xlsx')
#
# 集中竞价中的回购目的是否都有股票回购？
# a = pd.read_excel('info/buyback1.xls')
# b = pd.read_excel('info/buyback2.xls')
# c = pd.concat([a, b])
# def clean_illegal_characters(value):
#     if isinstance(value, str):
#         # 移除非法字符
#         return ''.join(c for c in value if ord(c) >= 32)
#     return value
# buyback1 = pd.read_excel('info/buyback1.xls')
# buyback2 = pd.read_excel('info/buyback2.xls')
# buyback = pd.concat([buyback1, buyback2], axis=0, ignore_index=True)
# print(buyback.info()
# print(pd.to_datetime('2018/10').strftime(
# ak.stock_zh_a_spot_em().to_excel('info/em.xlsx')
a = pd.read_excel('info/buyback.xls')[['A股股票代码_A_StkCd', '最新公司全称_LComNm']]
b = pd.read_excel('info/em.xlsx')

def vlidation(stock):
    if stock not in b.代码:
        return False
    else:
        return True
a['Is_true'] = a['A股股票代码_A_StkCd'].map(vlidation)
print(a['Is_true'].mean())
a.to_excel('info/a.xlsx')
a = pd.read_excel('visual_excel/cau_count.xlsx')
sns.lineplot(data=a, x='buyback_year', y='count')
plt.xticks(rotation=45)
plt.savefig('cau_count1.png')