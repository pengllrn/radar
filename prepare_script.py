# coding: utf-8
import pandas as pd
import numpy as np
import scipy.stats.stats as stats
from scipy.stats import chi2


#定义分箱函数（spearman）
def mono_bin(y,x,n=2):
    x2=x.fillna(np.median(x))
    r=0
    while np.abs(r)<0.8:
        d1=pd.DateFrame({"x":x2,"y":y,"bucket":pd.qcut(x2,n)})
        d2=d1.groupby("bucket",as_index=True)
        r,p=stats.spearmanr(d2.mean().x,d2.mean().y)
        n=n-1
    d3=pd.DateFrame(d2.min().x.tolist(),columns=["min_"+x.name])
    d3["max_"+x.name]=d2.max().x.tolist()
    d3[y.name]=d2.sum().y.tolist()
    d3["total"]=d2.count().y.tolist()
    d3[y.name+"_rate"]=d2.mean().y.tolist()
    d4=(d3.sort_index(by="min_"+x.name)).reset_index(drop=True)
    return d4
#mono_bin（data_try3.target,data_try3.id3_ap_mean)

###实际校验下来发现分出的箱不能通过一致检验（坏样本数量问题）

def chiMerge_maxInterval(chi_result,maxInterval=5):
    #卡方分箱合并————最大区间限制法
    group_cnt=len(chi_result)
    ##如果变量区间超过最大分箱限制，则根据合并原则合并
    while (group_cnt>maxInterval):
        min_index=chi_result[chi_result['chi_square']==chi_result["chi_square"].min()].index.tolist()[0]
        ##如果分箱区间在最前面，则向下合并
        if min_index==0:
            chi_result=merge_chiSquare[chi_result,min_index+1,min_index]
        ##如果分箱区间在最后，则向上合并
        elif min_index==group_cnt-1:
            chi_result=merge_chiSquare[chi_result,min_index-1,min_index]
        ##如果分箱区间在中间，则判断与其相邻的最小卡方区间，然后进行合并
        else:
            if chi_result.loc[min_index-1,'chi_square']>chi_result.loc[min_index+1,"chi_square"]:
                chi_result=merge_chiSquare(chi_result,min_index-1,min_index+1)
            else:
                chi_result=merge_chiSquare(chi_result,min_index-1,min_index)
        group_cnt=len(chi_result)
    return chi_result

#建立一个完整的卡方分布表
#再从这个表中取出（查找）想要的值
def get_chiSquare_distribution(dfree=4,cf=0.1):
    percents=[0.95,0.9,0.5,0.1,0.05,0.025,0.01,0.005]
##此处的意思：for循环遍历，将每次循环的结果作为参数，进行chi2.isf(percent,df)
    #chi2.isf()，计算卡方分布值。两个参数，分别是百分数，自由度
    #计算所有结果，保存为DataFrame
    df=pd.DataFrame(np.array([chi2.isf(percents,df=i) for i in range(1,30)]))
    df.columns=percents
    df.index=df.index+1
    ##显示小数点后面的数字，精确度
    pd.set_option('precision',3)
    return df.loc[dfree,cf]

def merge_chiSquare(chi_result,index,mergeIndex,a="expected_target_cnt",b="target_cnt",c="chi_square"):
    chi_result.loc[mergeIndex,a]=chi_result.loc[mergeIndex,a]+chi_result.loc[index,a]
    chi_result.loc[mergeIndex,b]=chi_result.loc[mergeIndex,b]+chi_result.loc[index,b]
    chi_result.loc[mergeIndex,c]=chi_result.loc[mergeIndex,c]+chi_result.loc[index,c]
    chi_result=chi_result.drop([index])
    chi_result=chi_result.reset_index(drop=True)
    return chi_result

#方法2  卡方统计量
#var为列名，sample是DataFrame数据集
def calc_chiSquare(var,sample):
    #计算卡方统计量
    ##计算样本期望频率
    #求和：即1样本的个数
    target_cnt=sample["target"].sum()
    #样本总数
    sample_cnt=sample["target"].count()
    ##期待的比率
    expected_ratio=target_cnt*1.0/sample_cnt
    #对样本按值大小进行排序，set去重
    df=sample[['var',"target"]]
    col_value=list(set(df[var]))
    col_value.sort()
    ##对变量区间进行遍历，计算每个区间对应的卡方统计量
    chi_list=[];target_list=[];expected_target_list=[]
    for value in col_value:
        df_target_cnt=df.loc[df[var]==value,"target"].sum()
        df_cnt=df.loc[df[var]==value,"target"].count()
        expected_target_cnt=df_cnt*expected_ratio
        ##卡方值，计算公式：（实际targert=1的样本数-理论target=1的样本数）的平方/理论target=1的样本数  累计和
        chi_square=(df_target_cnt-expected_target_cnt)**2/expected_target_cnt
        chi_list.append(chi_square)
        target_list.append(df_target_cnt)
        expected_target_list.append(expected_target_cnt)
    #导入结果到dataframe
    chi_result=pd.DataFrame({var:col_value,"chi_square":chi_list,"target_cnt":target_list,"expected_target_cnt":expected_target_list})
    return chi_result

def chiMerge_minChiSquare(chi_result,maxInterval=5):
    ###卡方分箱合并————卡方阈值法
    threshold=get_chiSquare_distribution(4,0.1)  #得到标准卡方分布值
    min_chiSquare=chi_result['chi_square'].min()
    group_cnt=len(chi_result)
    #如果变量区间的最小卡方值小于阈值，则继续合并直到最小值大于等于阈值
    while min_chiSquare<threshold and group_cnt>6:
        min_index=chi_result[chi_result['chi_square']==chi_result['chi_square'].min()].index.tolist()[0]
        ##如果分箱区间在最前面，则向下合并
        if min_index==0:
            chi_result=merge_chiSquare[chi_result,min_index+1,min_index]
        ##如果分箱区间在最后，则向上合并
        elif min_index==group_cnt-1:
            chi_result=merge_chiSquare[chi_result,min_index-1,min_index]
        ##如果分箱区间在中间，则判断与其相邻的最小卡方区间，然后进行合并
        else:
            if chi_result.loc[min_index-1,'chi_square']>chi_result.loc[min_index+1,"chi_square"]:
                chi_result=merge_chiSquare(chi_result,min_index-1,min_index+1)
            else:
                chi_result=merge_chiSquare(chi_result,min_index-1,min_index)
        group_cnt=len(chi_result)
    return chi_result

###计算woe
def calWOE(df,var,target):
    eps=0.00000001
    gbi=pd.crosstab(df[var],df[target])
    gb=df[target].value_counts
    gbri=gbi/gb
    gbri['woe']=np.log(gbri[1]/gbri[0])
    return gbri['woe'].tolist()

##计算IV
def calIV(df,var,target):
    eps=0.00000001
    gbi=pd.crosstab(df[var],df[target])
    gb=df[target].value_counts()
    gbri=gbi/gb
    gbri[var+'_target_count']=gbi[1]
    gbri[var+'_woe']=np.log(gbri[1]/gbri[0])
    gbri[var+'_iv']=(gbri[1]-gbri[0])*gbri[var+'_woe']
    gbri['iv_cum']=gbri[var+'_iv'].cumsum()
    return gbri

## 数据描述统计
def data_desc(data):
    (var_dat,max_dat,min_dat,nan_value_dat,nan_ratio_dat,median_dat,mode_dat,q10_dat,
     q90_dat)=([],[],[],[],[],[],[],[],[])
    for i in (data.columns.size):
        name=data.columns[i]
        max=data.iloc[:,i].max()
        min=data.iloc[:,i].min()
        nan_value=data.iloc[:,i].isnull().sum()
        nan_ratio=nan_value*1.0/len(data.iloc[:,i])
        median1=data.iloc[:,i].median()
        mode1=data.iloc[:,i].mode()[0]
        q10=data.iloc[:,i].quantile(0.1)
        q90=data.iloc[:,i].quantile(0.9)
        median_dat.append(median1)
        mode_dat.append(mode1)
        q10_dat.append(q10)
        q90_dat.append(q90)
        var_dat.append(name)
        max_dat.append(max)
        min_dat.append(min)
        nan_value_dat.append(nan_value)
        nan_ratio_dat.append(nan_ratio)
    dic={"var":var_dat,"max":max_dat,"min":min_dat,"nan_count":nan_value_dat,
         "nan_ratio":nan_ratio_dat,"median":median_dat,"mode":mode_dat,"q10":q10_dat,
         "q90":q90_dat}
    rst=pd.DataFrame(dic)
    return rst