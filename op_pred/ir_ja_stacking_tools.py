import re
import copy
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np

def sinlist2str(in_list):
    in_list_i_str = ''
    for i in in_list:
        in_list_i_str+=(str(i)+"    ")
    in_list_i_str = in_list_i_str[:-4]
    return in_list_i_str

def sinlist_str(combine_list):
    combine_str = ''
    for h in range(len(combine_list)):
        if h!=len(combine_list)-1:
            combine_str+=combine_list[h]+'_'
        else:
            combine_str+=combine_list[h]
    return combine_str
"""
https://weibo.com/1402400261/LmaAnCDH6?type=comment#_rnd1648737766546
"""
# def old_sinlist2sort_index(in_list,sort_para):
#     extra_item =[min(in_list)-abs(0.1*min(in_list)) if sort_para=='max2min' else max(in_list)+abs(0.1*max(in_list))][0]
#     sort_list = in_list+[extra_item]
#     sort_jd = 0
#     sort_index =[-1 for j in sort_list]
#     while sort_jd<len(in_list):
#         sort_tmp = [[extra_item if sort_index[j]!=-1 else sort_list[j]][0] for j in range(len(sort_list))]
#         # print(sort_tmp)
#         sort_curr_index = [sort_tmp.index(max(sort_tmp)) if sort_para=='max2min' else sort_tmp.index(min(sort_tmp))][0]
#         sort_index[sort_curr_index]=sort_jd
#         # print(sort_index)
#         sort_jd+=1
#         # print()
#     sort_index=sort_index[:-1]
    
#     return sort_index

def sinlist2sort_index(in_list,sort_para):
    # print('------------SORT START------------')
    sort_dict = {}
    for i in range(len(in_list)):
        sort_dict[i]=in_list[i]
    sorted_list = sorted(sort_dict.items(),key=lambda kv:(kv[1],kv[0]),reverse=[True if sort_para=='max2min' else False][0])
    ret_list = [-1 for i in in_list]
    for i in range(len(sorted_list)):
        ret_list[sorted_list[i][0]]=i
    # print('-------------SORT END-------------')
    return ret_list

# in_list = [1,2,3,4,5,6,2,0,-1,777,4,4,4,5]

# in_list = [0.111,0.236,-0.6,621539.526,0.559,-0.25941,0.000001,-12.60002]
# in_list = [0.6150948147179655,0.641677645010269,0.6635573446267352,0.6788490782957471,0.14829590595382436,-0.07353556847644205,0.48002947910356186,-6.398481186434389e+20,0.7467564792281931,0.5748966635788115,0.5491285201714687,0.5240724428234191,0.01520454238495983,0.5586165419892694,0.6258790616825156,0.5963787153699343,0.5232128805942204,0.15620440671248714,0.01803802637134655,-0.04815399837366274,0.3292838793313505,0.36700826792210406,-2.3026136215368873,0.2662882027635366,0.13522299208252198,0.3850868727040636,0.005971918466754822,0.18150885549083096,0.626154073055948,0.7437058678691677,0.6907831021721511,0.7504675475009446,0.09872052881474136,-0.07884785556183216,0.7695224163601159,-0.029062544805121382,0.6394392783338749,0.5576260038647711,0.5357281399273086,0.5235272637585421,0.24746998606728365,0.49578185499877214,0.6646396139895454,0.6664347961314234,0.6111030250435541,0.18279126390659906,0.13900638338838078,-0.06378943808498194,0.5878080179254436,0.5981989302804882,-0.5125385451863651,0.4555576954770517,0.23549334567665858,0.5800462325798112,0.010183529870098074,0.5072669217171979,0.699908226242932,0.7705327836891462,0.6913666550590236,0.7869684616764685,0.0874342209241109,-0.07846705389852682,0.8050002663774802,0.7698883655223195,0.7043641992139106,0.6152020337391306,0.5994310424478224,0.5891035705506149,0.27490662779746733,0.6158422693443579,0.6793134688870338,0.688941562944178,0.6869556981177842,0.265848558138459,0.13251719420508223,-0.09372122043188891,-0.19711805157296713,-0.197121002775013,-0.39404810824212966,0.5567366032525138,0.30100075064942156,0.5961247238190835,-0.007082673781100723,0.6401944571571112,0.6988030128644368,0.7419101739185485,0.6785387911613514,0.31253964332115913,0.1128776256419014,-0.09628857394259938,0.09900471070670747,0.07418377042962898,-0.4297479123126613,0.5457298870413426,0.3156591685188742,0.5786029921206799,-0.012537206525308875,0.622885519073532,0.6574694256603943,0.6045911737962755,0.6529143022884908,0.4300359832285554,0.13600046251911557,-0.07646447401427503,0.1889058980389492,-0.6117272120891719,-0.35720043988099603,0.6072703560687537,0.4109020024910316,0.6003713590438762,0.001362044030420173,0.6661380631008931,0.65981281411846,0.49291442396002894,0.6274389513081757,0.4546031029181341,0.15935576683368646,-0.0740583684259597,0.13829112294176227,-0.991159693358179,-0.4197612516607121,0.6184633821626986,0.534411637565181,0.5718353184791357,0.008740072710442748,0.6447234218082598,0.6175805637924647,0.6251380577779123,0.643191685038734,0.500347474472339,0.14069387693215885,-0.09288666207526997,0.15510114959199872,-85.95053697674808,0.4442072826807081,0.5555939500205012,0.5072803611085577,0.522820018729157,-0.008423686908400452,0.5694599533991903,0.7840703473295396,0.8193738597199489,0.8204805479050832,0.5580070685291112,0.009922466733204627,-0.09929029283974944,0.6983251255210748,-1.4673583361288496,0.0758426319707518,0.666580217853404,0.6299373525310488,0.6743300131416889,0.006087899744811276,0.6218024928991275,0.7093325797863581,0.7044619928402076,0.5771825200376934,0.7233403105632225,0.09523673569255908,-0.042468173800167985,0.6034551006130914,-30.46268176581177,0.44277281967524107,0.6811893819348469,0.7235385492265265,0.6296011653508199,0.06736971044606632,0.7274468259108995,0.8015914420222062,0.74846683637165,0.6687407916656152,0.47867233838708245,-0.03952746915921623,-0.1140026205211071,-0.31048916883991184,-40.268804782299206,0.5964513698202287,0.49687654955826344,0.5186451499022278,0.5333715953937809,-0.01423015108923531,0.572877236524426,0.8771585735056284,0.8367474830124554,0.8370120943122107,0.7198492023658545,0.08399445443066311,-0.08228979905955192,0.5985332129268186,-20.38749965891552,0.5655771206938704,0.7084052014725117,0.7102175802929651,0.6931071831996287,0.2639480509545642,0.6833168574537585,0.7097392990110469,0.7300791560861681,0.5594522411168572,0.5028244708348042,-0.0218733689609949,-0.10966293959629736,0.37320973345984654,-1.0379523424649508,0.5086899696303185,0.5201387708469354,0.5126646841529874,0.49925801704110107,0.11393688864661344,0.5686395735910982,0.7351806521685509,0.7114600515908557,0.6837554165653629,0.6719056439831759,0.110239685998748,-0.04737318025657622,0.6628375607911012,-3.685587531638042,0.32447557521033643,0.7095255881868578,0.678059260298642,0.6837595467316779,0.04908022210459606,0.7276654007172076,0.8076581021703237,0.7167004081943298,0.7657149604168783,0.5533703812875359,-0.007409377944029627,-0.09378024296775411,0.6928567226094096,-6.237084022371251,0.42881346617987404,0.6012165957872153,0.5812860487472795,0.6096064002886572,0.016758196410334714,0.5819813296129939,0.6845756706557073,0.6484645338335336,0.48716186322185184,0.46796275413764055,0.005554205692782532,-0.07803504739040523,-1.2204133024883927,-4.940606943586812,0.2328884527103241,0.5398993662806237,0.5400838205176992,0.5583848328546255,0.03278608904624547,0.4949085498079494,0.8054676499147057,0.8239357827005919,0.754441751467334,0.69951263308209,0.08725464413990869,-0.0678569972587415,0.7883553959535653,-5.049707326181956,0.5687604001433868,0.699114815614106,0.7019468158130977,0.6860552358025106,0.031330929900767024,0.6964534178511457,0.7664767517101462,0.696936495857717,0.7574869590858961,0.5671470520473261,0.01005549092966862,-0.08238993903879432,0.6001084075590828,-19.3099756431941,0.6601275643803803,0.5410292474251721,0.5234358841877447,0.5639114377735487,0.02418721380030653,0.5225890431102946,0.6405018230364314,0.6040637612645632,0.39884794058169404,0.3283376078668029,0.0774460241246544,-0.11004399625399497,0.34303066210900524,0.30485389886594416,-0.7593990021000556,0.6324149700799777,0.6197447209285559,0.6440399641661915,0.05204758421723954,0.37412027026170447,0.3975164094273419,0.34677376048169994,0.42419116360449205,0.005039854168830904,-0.01756945034829016,-0.1283963127624428,0.21773525498522162,0.17211117860108271,-1.114350326360214,0.09190422427550371,0.07129334082856431,0.04041413211866274,-0.03719117588718546,-0.13520988088457608,0.7384036632323168,0.8248032951049423,0.8251367014754943,0.8481914602426884,0.051229201716261996,-0.06416587143085906,0.746747131458332,0.7660644788477318,0.7220022696529296,0.7042443392665466,0.7147726572840413,0.6874235388101606,0.47610176125046033,0.7606258902986511,
# ]
# in_list = [-1000,-2000,-31241234123412,-3546354634634,-6745764]
# print(sinlist2sort_index(in_list,'max2min'))
# print(a(in_list,'max2min'))
# print("output:   ",a)
# # print("input:    ",in_list)
# for i in range(len(a)):
#     if not(i in a):
#         print(i)

def iscontain(short_str, long_str):
    short_re = re.split(r'_+', short_str)
    long_re = re.split(r'_+', long_str)
    # print(short_re,long_re)
    if not(short_re[0] in long_re):
        # print("?")
        return []
    long_index = 0
    ret_list_tmp = []
    while long_index <= len(long_re)-1:
        if long_re[long_index] == short_re[0]:
            ret_list_tmp.append([])
            short_index = 0
            while short_index <= len(short_re)-1:
                if long_index+short_index <= len(long_re)-1:
                    # print(long_index+short_index,short_re[short_index],long_re[long_index+short_index])
                    if short_re[short_index] == long_re[long_index+short_index]:
                        ret_list_tmp[-1].append(long_index+short_index)
                    else:
                        ret_list_tmp[-1].append("False")
                short_index += 1
            # long_index += short_index   210917  'mlr_ff_ori'  >>  'mlr_ff_mlr_ff_ori'
            long_index += 1
        else:
            long_index += 1
    ret_list_ret = []
    # print(ret_list_tmp)
    for i in ret_list_tmp:
        if not("False" in i):
            ret_list_ret.append(i)
    return ret_list_ret

def multi2printlines(input_sth,print_dict):
    """
    220301
    本代码存在的问题: 当某列仅含一个元素时，右侧列树状结构失效
    """
    # 相关需求见 ir_ja_stacking_doc.py
    error_name = 'multilist2printlines ERROR : '
    if not(input_sth) or not(print_dict):
        print(error_name+"input")
        print(input_sth,'\n',print_dict)
        return
    if print_dict['mode_select']=='floatright':
        if set(['col_n_int','col_contain_int','col_w_dict','col_title_dict']).difference(set(list(print_dict.keys()))):
            print(error_name+'print_dict[\'mode_floatright\'].keys()')
            return
        else:
            pass
    print_list = []
    # ?????????????
    if print_dict['col_contain_int']==0:
        for i in input_sth:
            print_list.append(str(i))
    print("print_list")
    print(print_list)
    print()
    if print_dict['col_n_int']==0:
        length_list =[]
        for i in print_list:
            length_list.append(len(str(i)))
        col_n_real = 1
        col_width = int(sum(length_list)/len(length_list))
        while col_width*col_n_real+2*(col_n_real-1)<=150:
            col_n_real+=1
        col_border_list = [2 for i in range(col_n_real-1)]
    col_title_list = []
    if print_dict['col_title_dict']:
        for i in print_dict['col_title_dict'].keys():
            if i=='0':
                if print_dict['col_title_dict']['0'][1]=='l':
                    col_title_list.append([print_dict['col_title_dict']['0'][0],0])
    
    ret_title_str=''
    for i in col_title_list:
        while len(ret_title_str)<i[1]:
            ret_title_str+=' '
        ret_title_str+=i[0]
    while len(ret_title_str)<=150:
        ret_title_str+=' '
    ret_list = [ret_title_str]
    for i in print_list:
        if len(ret_list[-1])+2+len(i)<=150:
            curr_line = copy.deepcopy(ret_list[-1])
        else:
            while len(ret_list[-1])<150:
                ret_list[-1]+=' '
            ret_list.append('')
            curr_line=''
        curr_col = '  '+i
        while len(curr_col)<col_width:
            curr_col+=' '
        curr_line+=curr_col
        # if len(i)>col_width:
        #     col_add_n = 1
        #     while col_width*col_add_n<len(i):
        #         col_add_n+=1
        #     for j in range(col_width*col_add_n-len(i)+1):
        #         curr_line+=' '
        ret_list[-1]=copy.deepcopy(curr_line)
    for i in range(len(ret_list)):
        if i!=0:
            ret_list[i]=ret_list[i][2:]
        
    return ret_list

# input_sth = [['dafa','daf','hrht'],['daf','daf',''],['dafa','daf','hrht'],['daf','daf',''],['adfadf','daf',''],['uiyo','uioy','6666'],['afaferw','',''],['daf','daf',''],['adfadf','daf',''],['asdfa','','afase'],['afaferw','',''],['daf','daf',''],['adfadf','daf',''],['asdfa','','afase']]
# print_dict = {
#     'mode_select':'floatright',
#         'col_n_int':0,
#         'col_contain_int':0,
#         'col_w_dict':{
#             'tot_list':[0],
#         },
#         'col_title_dict':{
#             '0':['the_content','l'],
#         },
    
# }
# for i in multi2printlines(input_sth,print_dict):
#     print(i)




# def sinlist2multidict(input_key_list,deepest_content):
#     for i in range(len(input_key_list)):
#         if i!=0:
#             curr_dict = {input_key_list[i]:deepest_content}
#             w_int = copy.deepcopy(i)-1
#             while w_int>=0:
#                 curr_dict = {input_key_list[w_int]:curr_dict}
#                 w_int= w_int-1
#     return curr_dict




def sinlist2multidict(input_key_list,deepest_content):
    curr_dict = {input_key_list[-1]:deepest_content}
    curr_int = len(input_key_list)-2
    while curr_int>=0:
        curr_dict = {input_key_list[curr_int]:curr_dict}
        curr_int = curr_int-1
    return curr_dict

# a = ['1','2','3','4','5','6']
# print(aa(a,{'a':111}))

# def aa(do,l,c):
#     curr_int = 0
#     do_l = []
#     while type({'?':'?'}) in [type(i) for i in do.values]:
#         curr_int+=1
#         for i in [i for i in do.keys() if ]
#         if curr_int==len(l):
# do = {1:{11:{111:'a'},22:{222:{2222:'b'}}},2:'c'}
# l = ['6']
# c = 'aaa'
# aa(do,l,c)


