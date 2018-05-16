import train
import os
import predict_result as pred
import get_next_data as gnt

params={
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'gamma':0.05,
    'max_depth':8,
    'lambda':4,
    'subsample':0.4,
    'colsample_bytree':0.6,
    'silent':1 ,
    'eta': 0.005,
    'seed':710,
    'nthread':4,
    'eval_metric':'error',
}

if __name__ == '__main__':
    FPPW_setted = 0.05
    Recall_target = 0.99
    FPPW =1
    Finnal_recall = Recall_target
    unit_count = 0
    data_path ='../unit_1/data_all/'
    log_lst = []
    unit_setting = open('./casced_setting.txt','w')
    while(FPPW>FPPW_setted):
        #do unit training
        train.train(data_path,params,unit_count)
        if unit_count ==0:
            data_path = data_path+os.sep+'train_data'
        out_ctx = pred.predict(data_path,unit_count)

        #get next_train_data
        out_path = './uinit_'+str(unit_count)+'_data'
        unit_thres,unit_fppw,unit_recall=gnt.run(out_ctx,data_path,Recall_target,out_path)

        #saving log
        log_lst.append(str(unit_thres)+' '+str(unit_fppw)+' '+str(unit_recall))
        print 'Finished '+str(unit_count)+' unit train:'
        print '[thres,fppw,recall]',log_lst[unit_count]
        write_ctx = str(unit_count)+' unit train: '+'[thres,fppw,recall] '+str(log_lst[unit_count])+'\n'
        unit_setting.write(write_ctx)

        ##upadate_input
        unit_count = unit_count+1
        data_path = out_path
        params['max_depth'] = params['max_depth']*2
        FPPW = FPPW*unit_fppw
        Finnal_recall = Finnal_recall * unit_recall

    print 'Finished all training... '+str(FPPW)+' '+str(Finnal_recall)

