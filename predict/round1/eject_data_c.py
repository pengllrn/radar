# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import prepare_script as ps

#读取数据，返回一个数据框
data_raw = pd.read_csv("F:\\data\\logistic\\round1\\com3.csv")
##iloc 区域选择 表示所有行，第3列以后
#drop  删除列（axis=1。  0为行），列名为：...
data_try1=data_raw.iloc[:,3:].drop(["subsys_num","dy_grp"],axis=1)

#获取数据描述，并保存为文件,不要写索引到文件
ada=ps.data_desc(data_try1)
ada.to_csv("F:\\data\\logistic\\round1\\data_desc.csv",index=False)

#对data_try1中的Nan值填充，每一列的填充值为每一列的中位数 median
#得到不包含缺失值的数据表：data_rty2
#保存文件
data_try2=data_try1.fillna(data_try1.median())
data_try2.to_csv("F:\\data\\logistic\\round1\\data_raw.csv",index=False)

##发射数据模型
#data_try2["Index"]==0返回一个与Index列等长的布尔值列表。当一个布尔值列表作为DataFrame的索引时，
#返回布尔值为True的行，并把所有的行组合成一个新的DataFrame
#Index=0表示发射，1表示接收
data_fs=data_try2[data_try2["Index"]==0].iloc[:,1:]
data_fs.to_csv("E:\\suqingsong\\logistic\\launch\\round1\\data_fs_try1.csv")
#计算每一列与其它列的相关性系数，并把计算结果保存为文件
corfs_1=data_fs.corr()
corfs_1.to_csv("E:\\suqingsong\\logistic\\launch\\round1\\corfs_1.csv")
#通过corfs_1.csv，可以发现有一些列存在空值。这说明这两列无法计算相关系数，或者完全无关
##去除无关列
list1=["gmk","gw","gzkb","zmO_bzgf1_abgz","zmO_bzgf1_abgz_ratio","zmO_bzgf1_srg1",
       "zmO_bzgf1_srg1_ratio"]

#等效于data_fs.drop(list1,axis=1)
for i in list1:
	data_fs=data_fs.drop(i,axis=1)
##删除了31个变量
#再次计算相关系数ρ
data_fs_try2=data_fs.iloc[:,1:]
corfs_2=data_fs_try2.corr()
corfs_2.to_csv("E:\\suqingsong\\logistic\\launch\\round1\\corfs_2.csv")
##去除相关性强的变量，和电源的平均电流和本振功放阵面是否过热过强相关，
# 阵面内发射前级的数据完全或负相关（拼接样本量少，颗粒度太大）


##特征工程
###卡方最优分箱（基于卡方阈值，本次分箱自由度为4，显著度0，01）
varlist=data_fs_try2.columns.tolist()[1:]  #列名集合
data_fs_tryx=data_fs_try2
for i in varlist:
	variance=i
	target="target"
	a=ps.chiMerge_minChiSquare(ps.calc_chiSquare(variance,data_fs_tryx),maxInterval=5)
	cutoffs=a[variance]

	data_fs_tryx["bining"+variance]=data_fs_tryx[variance].map(lambda x:value2grp(x,a[variance].tolist()))
	iv=ps.calIV(data_fs_tryx,"bining_"+variance,target)
	iv.to_csv("F:\\suqingsong\\logistic\\launch\\round1\\iv_r1.csv",index=False,mode="a")

#辅助函数
##将变量转化为相应的组
def value2grp(x,cutoffs):
	cutoffs=sorted(cutoffs)
	num_group=len(cutoffs)
	if x<cutoffs[0]:
		return a[variance][0]
	if x>cutoffs[-1]:
		return a[variance].tolist()[-1]
	for i in range(1,num_group):
		if cutoffs[i-1]<=x<cutoffs[i]:
			return a[variance][i]
		if x==cutoffs[i]:
			return a[variance][i]
list2=['zm0_bzgf1_scgl_ratio','zm0_bzgf2_scgl_ratio','zm0_bzgf2_zmgr_ratio','zm0_gr','zm0_gr_ratio']

data_fs_try2=data_fs_try2.drop(list2,axis=1)
##再次去除23个变量，目前剩下76个变量

#第二次相关性检验
corfs_3=data_fs.corr()
corfs_3.to_csv("F:\\suqingsong\\logistic\\launch\\round1\\corfs_3.csv")

##检查corfs_3，去掉没有关联的变量
#去掉30个变量
list3=['tdbh_ratio','cgwd_min','cgwd_max','fpga_max','id6_apavg_ma_3']
data_fs_try2=data_fs_try2.drop(list3,axis=1)

#保存文件
data_fs_try2.to_csv("F:\\suqingsong\\logistic\\launch\\round1\\fs_data1.csv",index=False)