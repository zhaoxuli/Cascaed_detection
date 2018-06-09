import train
import os
import get_best_score  as  gbs

Ada_params = {
    'dt_criterion':'gini',
    'dt_spliter':'random',
    'dt_max_depth':1,
    'dt_min_samples_split':10,
    'dt_max_features':'log2,',
    'dt_min_samples_leaf':5,
    'dt_max_leaf_nodes':40,
    'dt_class_weight':'balanced',

    'n_estimators':100,
    'learning_rate':0.7,
    'algorithm':'SAMME.R',
    'random_state':None
    }

if __name__ == '__main__':
    params = Ada_params
    FPPW_setted = 0.05
    Recall_target = 0.99
    FPPW =1
    Finnal_recall = Recall_target
    unit_count = 0
    data_path ='../data/'
    model_path = './models'
    info_recoder = './casced_setting.txt'

    if os.path.exists(model_path) ==False:
        os.makedirs(model_path)

    while(FPPW>FPPW_setted):
        #do unit training
        train_mat,pre_lst,label_lst = train.train(data_path,params,unit_count,model_path)
        if unit_count ==0:
            data_path = data_path+os.sep+'train_data'
        #get next_train_data
        out_path = './unit_'+str(unit_count)+'_data'
        if os.path.exists(out_path) ==False:
            os.makedirs(out_path)
        unit_thres,unit_fppw,unit_recall = gbs.gbs_run(pre_lst,label_lst,Recall_target,info_recoder,unit_count,train_mat,out_path)

        ##upadate_input
        unit_count = unit_count+1
        data_path = out_path
        params['dt_max_depth'] = params['dt_max_depth']+1
        FPPW = FPPW*unit_fppw
        Finnal_recall = Finnal_recall * unit_recall

    print 'Finished all training... '+str(FPPW)+' '+str(Finnal_recall)

