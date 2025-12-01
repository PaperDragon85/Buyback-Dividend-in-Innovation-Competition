import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from CAR import Stock, significance
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# def re_cha(text):
#     # 正则表达式移除所有控制字符
#     return re.sub(r'[\\x00-\\x1F\\x7F]', '', text)

buyback1 = pd.read_excel('info/buyback1.xls', dtype={'A股股票代码_A_StkCd': str})
buyback2 = pd.read_excel('info/buyback2.xls', dtype={'A股股票代码_A_StkCd': str})
buyback1 = buyback1.dropna(axis=1, how='all')
buyback2 = buyback2.dropna(axis=1, how='all')
buyback = pd.concat([buyback1, buyback2], axis=0, ignore_index=True)
info_df = buyback[[
    'A股股票代码_A_StkCd',
    '最新公司全称_LComNm',
    '首次信息发布日期_IInfoPubDt',
    '回购并注销股份公告书发布日_WrtOffPubDt',
    '事件进程描述_EventProcDes',
    '股份回购方式描述_RepoMtdDes',
    '回购目的说明_RepoPurDes'
]]
column_dict = {
    'A股股票代码_A_StkCd': 'stock_code',
    '最新公司全称_LComNm': 'name',
    '首次信息发布日期_IInfoPubDt': 'first_info',
    '回购并注销股份公告书发布日_WrtOffPubDt': 'first_buyback',
    '事件进程描述_EventProcDes': 'completed',
    '股份回购方式描述_RepoMtdDes': 'way',
    '回购目的说明_RepoPurDes': 'purpose',
}
info_df = info_df.rename(columns=column_dict)

# 检查数据情况
# for i in info_df.columns:
#     print(f'{i}列unique数量为{info_df[i].nunique()}')
# 生成年份变量
def year(date):
    if date:
        return str(date.year)
    else:
        print('有缺失值')

# info_df含有成功回购和未成功回购的数据都有
info_df['info_year'] = info_df['first_info'].map(year)
info_df['buyback_year'] = info_df['first_buyback'].map(year)


def cau_count(df, date_type='buyback_year', excel=False, plot=False):
    df['count'] = 0
    df = df[(df['completed'] == '实施完成') & (df['buyback_year'].notna())].sort_values(by=date_type, ascending=False)
    grouped_df = df.groupby(date_type)['count'].count().reset_index()
    grouped_df = grouped_df.dropna()
    for i in grouped_df['buyback_year']:
        print(i, type(i))
    if excel == True:
        grouped_df.to_excel('visual_excel/cau_count.xlsx')
    if plot == True:
        sns.lineplot(data=grouped_df, x=date_type, y='count')
        plt.title('每年总回购数')
        plt.xticks(rotation=45)
        plt.savefig(f'visual_plot/cau_count.png')
    return grouped_df

# 集中竞价占总回购数
def focus(text):
    if text == '集中竞价':
        return True
    else:
        return False
def ratio(series):
    return series.sum() / series.count()

def cau_positive(df, date_type='buyback_year', excel=False, plot=False):
    df = df[df['completed'] == '实施完成'].sort_values(by=date_type, ascending=False)
    df['count'] = df['way'].map(focus)
    grouped_df = df.groupby(date_type)['count'].agg(ratio).reset_index()
    grouped_df[date_type] = grouped_df[date_type].astype(int).astype(str)
    if excel == True:
        grouped_df.to_excel('visual_excel/cau_positive.xlsx')
    if plot == True:
        sns.barplot(data=grouped_df, x=date_type, y='count', color='green', label='集中竞价')
        plt.xticks(rotation=45)
        plt.savefig('visual_plot/cau_positive')
    return grouped_df

def positive_word(text, word_list):
    for i in word_list:
        if i in text:
            return True
    else:
        return False

# 集中竞价的回购的目的中的关键词
def cau_purpose(df, word_list, date_type='buyback_year', excel=False, plot=False):
    df = df[df['first_buyback'] != pd.NA]
    df = df[df['completed'] == '实施完成'].sort_values(by=date_type, ascending=False)
    df = df[df['way'] == '集中竞价']
    df['count'] = df['purpose'].map(lambda x: positive_word(x, word_list))
    df = df.dropna(subset='buyback_year')
    df.to_excel('df.xlsx')
    grouped_df = df.groupby(date_type)['count'].agg(ratio).reset_index()
    if excel == True:
        grouped_df.to_excel('visual_excel/cau_purpose.xlsx')
    if plot == True:
        sns.barplot(data=grouped_df, x=date_type, y='count', color='green', label=f'{word_list}')
        plt.xticks(rotation=45)
        plt.savefig(f'visual_plot/cau_purpose{word_list}')
    return grouped_df

# 减少循环任务
# new_info_df = info_df.dropna(subset=['buyback_year'])
new_info_df = info_df[info_df['way'] == '集中竞价'].dropna(subset=['buyback_year'])
print(new_info_df.dtypes)
#获取某年平均超额收益
def mean_car(year, sample=None, date_type='buyback_year', window_time=10, alpha=0.05, excel=False, plot=False):
    year = str(year)
    if date_type == 'buyback_year':
        date_name = 'first_buyback'
    else:
        date_name = 'first_info'
    if sample == None:
        sample_df = new_info_df[new_info_df[date_type] == year]
    else:
        counts = len(new_info_df[new_info_df[date_type] == year])
        if counts < sample:
            sample_df = new_info_df[new_info_df[date_type] == year].sample(n=counts, replace=False)
        else:
            sample_df = new_info_df[new_info_df[date_type] == year].sample(n=sample, replace=False)
    sample_list = []
    significance_count = 0
    for index, row in sample_df.iterrows():
        stock = row['stock_code']
        handle_time=row[date_name].strftime('%Y-%m-%d')
        try:
            a = Stock(stock, handle_time, window_time=window_time)
            b = a.ols()
            if b['CAR_p'].iloc[0] < alpha:
                significance_count += 1
            sample_list.append(b)
        except:
            continue
    sample_n = len(sample_list)
    aar = pd.concat([i.AR for i in sample_list], axis=1)
    aar = aar.agg('mean', axis=1)
    result_df = pd.DataFrame({'AAR': aar})
    result_df = result_df.reset_index()
    result_df['index'] = result_df['index'] - window_time
    result_df['CAR'] = result_df['AAR'].cumsum()
    result_df['significance'] = significance_count
    result_df['sample'] = sample_n
    result_df['significance_ratio'] = result_df['significance'] / result_df['sample']
    print(result_df)
    print(sample_n)
    if excel:
        result_df.to_excel(f'visual_excel/mean_car_{year}.xlsx')
    if plot:
        sns.lineplot(data=result_df, x='index', y='AAR', label='AAR')
        sns.lineplot(data=result_df, x='index', y='CAR', label='CAR')
        plt.xticks(rotation=45)
        plt.title('集中竞价回购平均超额收益')
        plt.savefig(f'visual_plot/mean_car_{year}.png')
    return result_df


def year_loop(first_year=2010, last_year=2025, window_time=10, alpha=0.05, sample=None, excel=False, plot=False):
    year_list = []
    for year in [str(year) for year in range(first_year, last_year)]:
        result_df = mean_car(year, sample, window_time=window_time, alpha=alpha, excel=excel)
        p = result_df['significance_ratio'].iloc[0]
        year_list.append((year,p))
    year_df = pd.DataFrame(year_list, columns=['year', 'significance_ratio'])
    print(year_df)
    if excel:
        year_df.to_excel('year_data/year_df.xlsx')
    if plot:
        sns.barplot(data=year_df, x='year', y='significance_ratio')
        plt.savefig('year_data/year_significance.png')
    return year_df
if __name__ == '__main__':
    # mean_car('2016', 10, date_type='info_year', excel=True, plot=True)
    # year_loop(2015, 2025, window_time=15, excel=True, plot=True)
    # cau_purpose(info_df, ['信心'], date_type='buyback_year', excel=True, plot=True)
    # cau_count(info_df, excel=True, plot=True)
    mean_car(2023, date_type='info_year', sample=10, window_time=15, excel=True, plot=True)