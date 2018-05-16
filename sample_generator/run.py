import  generator as gner


parameters={
    'ref_type':'height',
    'ref_h':60,
    'ref_w':40,
    'norm_len':58,
    'resize':256,
    'min_vail_len':10,
    'positive_up_boundary' :1.1,
    'positive_low_boundary':0.9,
    'regression_up_boundary' :1.1,
    'regression_low_boundary':0.9,
    'ignore_up_boundary':2,
    'ignore_low_boundary':0.5,
    'postive_gener_ratio'  :0.1,
    'regression_gener_ratio':0.2,
    'ignore_gener_ratio':0.4,
    'Image_path':'/mnt/hgfs/linux_share/phone_data/Image',
    'Anno_path' :'/mnt/hgfs/linux_share/phone_data/Anno_txt'
}


gner.run(parameters)
