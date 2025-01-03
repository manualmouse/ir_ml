from numpy.core.numeric import cross
from pandas.io.sql import DatabaseError
from sklearn.model_selection import train_test_split,GridSearchCV,LeaveOneOut,cross_val_score,learning_curve,ShuffleSplit,StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import Lasso,LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample,shuffle
import lightgbm as lgb

import pandas as pd
import numpy as np
import copy
import re
import os
import time
import math
from multiprocessing import Pool, log_to_stderr

from ir_ja_stacking_tools import sinlist2str
from ir_ja_stacking_tools import sinlist_str
from ir_ja_stacking_tools import sinlist2sort_index
from ir_ja_stacking_tools import iscontain

"""
本脚本提供 在本地 重跑相关 的代码
"""

class LagReadAndRun(object):
    def __init__(self,typical_path,multimlr_path,log_path_list):

        # 注意 log 顺序 : 后面的覆盖前面的


        
        

        self.typical_path = typical_path
        self.multimlr_path = multimlr_path
        self.tot_dict = {}
        self.ret_log_list = []
        self.log_path_list = log_path_list

        ty_o = open(self.typical_path,'r',encoding='utf-8')
        mm_o = open(self.multimlr_path,'r',encoding='utf-8')
        self.ret_log_list.append(self.typical_path)
        self.ret_log_list.append(self.multimlr_path)
        ty_rdl = ty_o.readlines()
        mm_rdl = mm_o.readlines()
        tot_rdl = list(ty_rdl)+list(mm_rdl)
        tot_index = 0
        while tot_index<=len(tot_rdl)-1:
            tot_title = re.split(r'\s+',tot_rdl[tot_index])
            if len(tot_title[:-1])==8:
                tot_title = tot_title[:-1]
                if re.findall(r'dl1\S+',tot_title[2]) and re.findall(r'dl2\S+',tot_title[3]) and (tot_title[-1]=='part' or tot_title[-1]=='all'):
                    if tuple(tot_title) in self.tot_dict:
                        print("ERROR    tuple(tot_title) in self.tot_dict")
                        print(tot_title)
                        print(self.tot_dict[tot_title])
                        print("old---------------------------new")
                        for i in range(1,8):
                            print(tot_rdl[tot_index+i])
                    # elif (
                    #     (('rg_07f' in tot_title) or ('tot_emgauss_lambda' in tot_title) or ("tot_lifetime" in tot_title) or ("tot_sigma" in tot_title))
                    #     and ('mlr_ff_final' in tot_title)
                    #     ):
                    #         pass
                    else:
                        self.tot_dict[tuple(tot_title)]=[]
                        for i in range(1,8):
                            self.tot_dict[tuple(tot_title)].append(re.split(r'\s+',tot_rdl[tot_index+i])[0])
                        self.tot_dict[tuple(tot_title)]+=[[i for i in re.split(r'\s+',tot_rdl[tot_index+8])[:-1] if i]]
                tot_index+=9
            else:
                tot_index+=1


        self.tot_position_filter = [[]for i in range(len(list(self.tot_dict.keys())[0]))]
        for i in self.tot_dict:
            for j in range(len(i)):
                if not(i[j] in self.tot_position_filter[j]):
                    self.tot_position_filter[j].append(i[j])
        # for i in self.tot_position_filter:
        #     print(i)

    def show_tot_pfilter(self):
        tot_pfliter_max_len_list = []
        tot_pfliter_num_list = []
        for i in self.tot_position_filter:
            tot_pfliter_max_len_list.append(0)
            tot_pfliter_num_list.append(len(i))
            for j in i:
                if len(j)>tot_pfliter_max_len_list[-1]:
                    tot_pfliter_max_len_list[-1]=len(j)
        print("def show_tot_pfilter(self)")
        def space_num(s_num):
            ret_space = ''
            for i in range(s_num):
                ret_space+=' '
            return ret_space
        show_filter_num_str = ''
        for i in range(len(self.tot_position_filter)):
            show_filter_num_str+=(
                str(tot_pfliter_num_list[i])+\
                space_num(tot_pfliter_max_len_list[i]-len(str(tot_pfliter_num_list[i])))
                )
            if i!=len(self.tot_position_filter)-1:
                show_filter_num_str+='    '
        print(show_filter_num_str)
        for i in range(max(tot_pfliter_num_list)):
            show_currl = ''
            for j in range(len(self.tot_position_filter)):
                if i>len(self.tot_position_filter[j])-1:
                    show_currl+=space_num(tot_pfliter_max_len_list[j])
                else:
                    show_currl+=(
                        self.tot_position_filter[j][i]+\
                            space_num(tot_pfliter_max_len_list[j]-len(str(self.tot_position_filter[j][i])))
                        )
                if j!=len(self.tot_position_filter)-1:
                    show_currl+='    '
            print(show_currl)
        print("\n\n")

    def dict_filter(self,and_list,or_list,not_list,chongfu_int):


        self.all_lag_dict = {}
        for i in self.log_path_list:
            log_o = open(i,'r',encoding='utf-8')
            log_rdl = log_o.readlines()
            log_fname_index = 0
            while log_fname_index<len(log_rdl):
                curr_fname_re = re.split(r'\s+',log_rdl[log_fname_index])[:-1]
                if len(curr_fname_re)>len(and_list) and set(and_list)<set(curr_fname_re):
                    # print("log_fname_index    "+str(log_fname_index))
                    log_method_index = copy.deepcopy(log_fname_index)+1
                    while(
                        log_method_index<len(log_rdl)
                        and not(set(re.split(r'\s+',log_rdl[log_method_index])[:-1])>set(and_list))
                    ):
                        if 'for j in learner_tot_dict' in log_rdl[log_method_index]:
                            # print("log_method_index    "+str(log_method_index))
                            curr_method_re = re.split(r'\s+',log_rdl[log_method_index])[0]
                            self.all_lag_dict[tuple(curr_fname_re+[curr_method_re])]={}
                            log_result_e_index = copy.deepcopy(log_method_index)+1
                            while(
                                log_result_e_index<len(log_rdl)
                                and not(set(re.split(r'\s+',log_rdl[log_result_e_index])[:-1])>set(and_list))
                                and not('for j in learner_tot_dict' in log_rdl[log_result_e_index])
                            ):
                                curr_result_e_re = re.split(r'\s+',log_rdl[log_result_e_index])[:-1]
                                if len(curr_result_e_re)==4 and curr_result_e_re[1]=='every_rmse' and curr_result_e_re[3]=='every_r2':
                                    # print("log_result_e_index    "+str(log_result_e_index))
                                    self.all_lag_dict[tuple(curr_fname_re+[curr_method_re])]['every_rmse']= float(curr_result_e_re[0])
                                    self.all_lag_dict[tuple(curr_fname_re+[curr_method_re])]['every_r2']= float(curr_result_e_re[2])
                                    if '>>>>  last curr_p_and_r_list  <<<<' in log_rdl[log_result_e_index-3]:
                                        log_result_train_index = copy.deepcopy(log_result_e_index-2)
                                    else:
                                        log_result_train_index = copy.deepcopy(log_result_e_index-3)
                                    # print(log_rdl[log_result_train_index])
                                    curr_result_train_re1 = re.findall(r'\[\{.*\},(.+)\]\s+',log_rdl[log_result_train_index])[0]
                                    curr_result_train_re2 = [j for j in re.split(r'[\s+,\'\"]',curr_result_train_re1) if j]
                                    self.all_lag_dict[tuple(curr_fname_re+[curr_method_re])]['train_rmse']=abs(float(curr_result_train_re2[-2]))
                                    self.all_lag_dict[tuple(curr_fname_re+[curr_method_re])]['train_r2']=abs(float(curr_result_train_re2[-1]))
                                    self.all_lag_dict[tuple(curr_fname_re+[curr_method_re])]['para_dict']={}
                                    if len(curr_result_train_re2)>2:
                                        curr_para_list = curr_result_train_re2[:-2]
                                        for j in curr_para_list:
                                            curr_result_train_re3 = re.split(r':',j)
                                            # print(curr_result_train_re3)
                                            if '.' in curr_result_train_re3[1] or 'e' in curr_result_train_re3[1]:
                                                curr_result_train_para = float(curr_result_train_re3[1])
                                            else:
                                                curr_result_train_para = int(curr_result_train_re3[1])
                                            self.all_lag_dict[tuple(curr_fname_re+[curr_method_re])]['para_dict'][curr_result_train_re3[0]]=[copy.deepcopy(curr_result_train_para)]
                                log_result_e_index+=1
                            log_method_index=copy.deepcopy(log_result_e_index)
                        else:
                            log_method_index+=1
                    log_fname_index=copy.deepcopy(log_method_index)
                else:
                    log_fname_index+=1
        
        for i in self.all_lag_dict:
            print(i)
            for j in self.all_lag_dict[i]:
                print(j+'    '+str(type(self.all_lag_dict[i][j]))+"    "+str(self.all_lag_dict[i][j]))
            print()


        self.dict_filter_ret_list = []
        select_list = []
        skip_list = []
        x_columns_num_dict  ={}
        for i in self.tot_dict:
            curr_nand_num = 0
            curr_or_num = 0
            curr_not_num = 0
            if and_list:
                for j in and_list:
                    if not(j in i):
                        curr_nand_num = 1
                        break
            if or_list:
                for j in or_list:
                    if j in i:
                        curr_or_num=1
                        break
            else:
                curr_or_num=1
            if not_list:
                for j in not_list:
                    if j in i:
                        curr_not_num+=1
                        break
            if not(curr_nand_num) and curr_or_num and not(curr_not_num):
                select_list.append(i)

            # print(self.tot_dict[i])
            curr_x_col_len = len(self.tot_dict[i][-1])
            if not(curr_x_col_len in x_columns_num_dict):
                x_columns_num_dict[curr_x_col_len]=[]
            x_columns_num_dict[curr_x_col_len].append([i,self.tot_dict[i][-1]])
        
        # print(x_columns_num_dict.keys())
        # for i in range(3):
        # # for i in range(5):
        #     if not(i in x_columns_num_dict):
        #         continue
        #     print(i)
        #     for j in x_columns_num_dict[i]:
        #         print(j[0])
        #         print(j[1])
        #         print()
        #     print()
        quchong_list = ['des_corr','des_rf_ff','des_la_ff','des_mlr_ff','des_random']
        if chongfu_int:
            for i in x_columns_num_dict:
                for j in range(len(x_columns_num_dict[i])):
                    if i==0:
                        skip_list.append(x_columns_num_dict[i][j][0])
                        continue
                    for k in range(len(x_columns_num_dict[i])):
                        # print(list(x_columns_num_dict[i][j][0])[:5]+[list(x_columns_num_dict[i][j][0])[-1]])
                        if (
                            (k>j) 
                            and ((list(x_columns_num_dict[i][j][0])[:5]+[list(x_columns_num_dict[i][j][0])[-1]])==(list(x_columns_num_dict[i][k][0])[:5]+[list(x_columns_num_dict[i][k][0])[-1]])) 
                            and set(x_columns_num_dict[i][j][1])==set(x_columns_num_dict[i][k][1])
                            ):
                            
                            # print("    ",x_columns_num_dict[i][j],'\n',"    ",x_columns_num_dict[i][k],'\n')
                            quchong_j_index = len(quchong_list)
                            quchong_k_index = len(quchong_list)
                            for l in range(len(quchong_list)):
                                if quchong_list[l] in x_columns_num_dict[i][j][0]:
                                    quchong_j_index=l
                                if quchong_list[l] in x_columns_num_dict[i][k][0]:
                                    quchong_k_index=l
                            # print(x_columns_num_dict[i][j][0],x_columns_num_dict[i][k][0])
                            if quchong_j_index>quchong_k_index:
                                skip_list.append(x_columns_num_dict[i][j][0])
                            else:
                                skip_list.append(x_columns_num_dict[i][k][0])
        for i in select_list:
            if not(i in skip_list):
                self.dict_filter_ret_list.append(i)

        for i in self.dict_filter_ret_list:
            print(i)
            for j in range(len(self.tot_dict[i])):
                if j!=len(self.tot_dict[i])-1:
                    print(self.tot_dict[i][j])
                else:
                    if len(self.tot_dict[i][j])>20:
                        print(len(self.tot_dict[i][j]))
                    else:
                        print(self.tot_dict[i][j])
            print()

        print(len(self.dict_filter_ret_list))
        return self.dict_filter_ret_list

    def learner_a_global(self,learner_list,output_dir):
        def learner_ret(reg_str,kernel_str):
            if kernel_str:
                if reg_str=='svm':
                    return SVR(kernel=kernel_str)
                elif reg_str=='krr':
                    return KernelRidge(kernel=kernel_str)
                else:
                    print("ERROR    reg_str==?")
            else:
                return learner_reg_dict[reg_str]
        if not(learner_list):
            print("ERROR    if not(learner_list)")
            return
        if 1:
            learner_reg_dict={
                'rf':RandomForestRegressor(random_state=42),
                'gbrt':GradientBoostingRegressor(random_state=42),
                'lgbm':lgb.LGBMRegressor(),
                'la':Lasso(random_state=42,max_iter=100000),
                'mlr':LinearRegression(),
            }
            learner_cv_dict={
                'shu':ShuffleSplit(n_splits=5,train_size=0.8,test_size=0.2,random_state=24),
            }

        if learner_list==['all']:
            learner_tot_dict={
                'rf': ['n_estimators', 'max_depth','max_features',
                'max_leaf_nodes'
                ],
                'gbrt': ['n_estimators', 'learning_rate','max_depth','max_features','max_leaf_nodes'],
                'lgbm': ['n_estimators', 'learning_rate','max_depth', 'num_leaves'],
                'svm_linear': [],
                'svm_rbf': ['gamma'],
                'svm_sigmoid': ['gamma','coef0'],
                'la': ['alpha'],
                # 'krr_linear': [],
                # 'krr_rbf': ['gamma'],
                # 'krr_laplacian': ['gamma'],
                # # 'krr_sigmoid': ['gamma','coef0'],
                # 'krr_cosine': [],
            }
        else:
            return 
        print("\n\n------------  def learner_a_global(self,learner_list)  ------------\n\n")
        self.learner_a_ret_dict ={}
        lag_log_o = open(re.findall(r'(\S+)job_list_typical_\S*\.txt',self.typical_path)[0]+"lag_cre_"+time.strftime("%Y%m%d%H%M%S", time.localtime())+'.txt','w',encoding='utf-8')
        for i in self.dict_filter_ret_list:
            print(i,"    for i in self.dict_filter_ret_list")
            lag_log_o.write(sinlist2str(list(i))+"\n")
            x_test_ori = pd.read_csv(self.tot_dict[i][1],index_col=[0]).loc[:,self.tot_dict[i][7]]
            y_test_ori = pd.read_csv(self.tot_dict[i][2],index_col=[0]).values.ravel()
            x_test_re = pd.read_csv(self.tot_dict[i][3],index_col=[0]).loc[:,self.tot_dict[i][7]]
            y_test_re = pd.read_csv(self.tot_dict[i][4],index_col=[0]).values.ravel()
            x_train_rew = pd.read_csv(self.tot_dict[i][5],index_col=[0]).loc[:,self.tot_dict[i][7]]
            y_train_rew = pd.read_csv(self.tot_dict[i][6],index_col=[0]).values.ravel()
            if not(tuple(list(i)[:5]+[list(i)[-1]]) in self.learner_a_ret_dict):
                self.learner_a_ret_dict[tuple(list(i)[:5]+[list(i)[-1]])]={}
                self.learner_a_ret_dict[tuple(list(i)[:5]+[list(i)[-1]])]['y_test_re']=[pd.read_csv(self.tot_dict[i][4],index_col=[0])]
                self.learner_a_ret_dict[tuple(list(i)[:5]+[list(i)[-1]])]['y_test_ori']=[pd.read_csv(self.tot_dict[i][2],index_col=[0])]
            self.ret_log_list.append(sinlist2str(i))
            for j in learner_tot_dict:
                print(j,'    for j in learner_tot_dict')
                lag_log_o.write(str(j)+'    for j in learner_tot_dict\n')
                print(learner_tot_dict[j],"    learner_tot_dict[j]")
                # lag_log_o.write(sinlist2str(list(learner_tot_dict[j]))+"    learner_tot_dict[j]\n")
                reg_str = re.split(r'_+',j)[0]
                if len(re.split(r'_+',j))==2:
                    kernel_str = re.split(r'_+',j)[1]
                else:
                    kernel_str=''


                print()
                if learner_tot_dict[j]:
                    m_f = GridSearchCV(
                        learner_ret(reg_str,kernel_str),
                        param_grid=self.all_lag_dict[tuple(list(i)+[j])]['para_dict'],
                        cv=learner_cv_dict['shu'],
                        scoring='neg_root_mean_squared_error',
                        n_jobs=-1
                    ).fit(x_train_rew,y_train_rew)
                    curr_estimator = m_f.best_estimator_
                    curr_rmse = float(np.sqrt(mean_squared_error(y_train_rew,curr_estimator.predict(x_train_rew))))
                    curr_r2 = float(r2_score(y_train_rew,curr_estimator.predict(x_train_rew)))
                    lag_log_o.write(str(self.all_lag_dict[tuple(list(i)+[j])]['para_dict'])+'\n')
                    lag_log_o.write(str(curr_rmse)+"    curr_rmse    "+str(curr_r2)+"    curr_r2\n")
                else:
                    m_f=learner_ret(reg_str,kernel_str)
                    m_f.fit(x_train_rew,y_train_rew)
                    curr_estimator = m_f
                    curr_rmse = float(np.sqrt(mean_squared_error(y_train_rew,curr_estimator.predict(x_train_rew))))
                    curr_r2 = float(r2_score(y_train_rew,curr_estimator.predict(x_train_rew)))
                    lag_log_o.write("{}, "+str(curr_rmse)+"    curr_rmse    "+str(curr_r2)+"    curr_r2\n")

                every_re_predict = curr_estimator.predict(x_test_re)
                every_re_rmse = float(np.sqrt(mean_squared_error(y_test_re,every_re_predict)))
                every_re_r2 = float(r2_score(y_test_re,every_re_predict))
                every_re_ppd = pd.DataFrame(
                    index=x_test_re.index.tolist(),
                    data=every_re_predict,
                    columns = [i[5]+'_'+i[6]+'_'+reg_str+['_'+kernel_str if kernel_str else ''][0]+'_'+str(every_re_rmse)+"_"+str(every_re_r2)]
                )
                lag_log_o.write(str(every_re_rmse)+"    every_re_rmse    "+str(every_re_r2)+'   every_re_r2\n')



                every_ori_predict = curr_estimator.predict(x_test_ori)
                every_ori_rmse = float(np.sqrt(mean_squared_error(y_test_ori,every_ori_predict)))
                every_ori_r2 = float(r2_score(y_test_ori,every_ori_predict))
                every_ori_ppd = pd.DataFrame(
                    index=x_test_ori.index.tolist(),
                    data=every_ori_predict,
                    columns = [i[5]+'_'+i[6]+'_'+reg_str+['_'+kernel_str if kernel_str else ''][0]+'_'+str(every_ori_rmse)+"_"+str(every_ori_r2)]
                )
                lag_log_o.write(str(every_ori_rmse)+"    every_ori_rmse    "+str(every_ori_r2)+'   every_ori_r2\n')




                self.learner_a_ret_dict[tuple(list(i)[:5]+[list(i)[-1]])][tuple(list(i)[5:-1]+[reg_str,kernel_str])]={
                    'x_test_ori':[
                        copy.deepcopy(every_ori_rmse),
                        copy.deepcopy(every_ori_r2),
                        copy.deepcopy(every_ori_ppd)],
                    'x_test_re':[
                        copy.deepcopy(every_re_rmse),
                        copy.deepcopy(every_re_r2),
                        copy.deepcopy(every_re_ppd)]
                }
                lag_log_o.write('\n\n\n')








        for i in self.learner_a_ret_dict:
            ret_ori_pd = pd.DataFrame()
            ret_re_pd = pd.DataFrame()
            for j in self.learner_a_ret_dict[i]:
                if j =='y_test_re':
                    ret_re_pd= self.learner_a_ret_dict[i][j][-1]
                elif j =='y_test_ori':
                    ret_ori_pd= self.learner_a_ret_dict[i][j][-1]
                else:
                    self.learner_a_ret_dict[i][j]['x_test_ori'][-1].index=ret_ori_pd.index.tolist()
                    ret_ori_pd = pd.concat([ret_ori_pd,self.learner_a_ret_dict[i][j]['x_test_ori'][-1]],axis=1,sort=False)

                    self.learner_a_ret_dict[i][j]['x_test_re'][-1].index=ret_re_pd.index.tolist()
                    ret_re_pd = pd.concat([ret_re_pd,self.learner_a_ret_dict[i][j]['x_test_re'][-1]],axis=1,sort=False)



            ret_str='_'.join(i)
            ret_ori_pd.T.to_csv(output_dir+'/pred_rr/'+"T_"+ret_str+"_"+time.strftime("%Y%m%d%H%M%S", time.localtime())+".csv",index=True,header=True)
            ret_re_pd.T.to_csv(output_dir+'/pred_cre/'+"T_"+ret_str+"_"+time.strftime("%Y%m%d%H%M%S", time.localtime())+".csv",index=True,header=True)

