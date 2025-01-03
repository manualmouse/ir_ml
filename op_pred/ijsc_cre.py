from ir_ja_stacking_cre import LagReadAndRun
des_name_list = [
            'fr_4_s_n',
    ]



sla_obj = LagReadAndRun(
    './ir_weirui24/snresult_slf/job_list_typical_.txt',
    './ir_weirui24/snresult_slf/job_list_multimlr_.txt',
    ['./ir_weirui24/snresult_slf/lag_20240512180728.txt']
)
sla_obj.show_tot_pfilter()
sla_obj.dict_filter(
    ['slf','dl1ss','dl2ss','all'],
    des_name_list,
    ['part','mlr_ff_ori'],1)
sla_obj.learner_a_global(['all'],'./ir_weirui24/snresult_slf')
    

# import json
# final_dict = dict()
# for i in sla_obj.dict_filter_ret_list:
#     nk = '~'.join(i)
#     if not(nk in final_dict):
#         final_dict[nk]=[dict(),list()]
#     final_dict[nk][1]=sla_obj.tot_dict[i]
# for i in sla_obj.all_lag_dict:
#     final_dict['~'.join(i[:-1])][0][i[-1]]=sla_obj.all_lag_dict[i]

# curr_name = './ir_weirui24/model.json'
# json_str = json.dumps(final_dict)
# json_file = open(curr_name,'w')
# json_file.write(json_str)