### Xgboost_Casced_Classifier    Document
`data_process.py` 根据sample_generator 生成的正例和反例样本获得，`train_anno.txt`&`test_anno.txt`   
`get_npy.py` change the train & test image to npy type for train,because the binary brffer is too big for xgboost   
`run.py`  is  a summary script used to train,include	
* `train.py` to train each xgboost model 
* `predict_result.py`to get each model's  result   on train_data. 
* `get_next_data.py`to get best thres and next train_data by predict result. 

`predict_casced_model.py`predict all data with all model  by caseceding  way.