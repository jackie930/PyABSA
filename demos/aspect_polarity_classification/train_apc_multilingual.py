# -*- coding: utf-8 -*-
# file: train_apc_multilingual.py
# time: 2021/5/26 0026
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                    train and evaluate on your own apc_datasets (need train and test apc_datasets)                    #
########################################################################################################################


from pyabsa.functional import Trainer
from pyabsa.functional import APCConfigManager
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import APCModelList

apc_config_multilingual = APCConfigManager.get_apc_config_multilingual()
apc_config_multilingual.model = APCModelList.FAST_LCF_BERT
apc_config_multilingual.evaluate_begin = 3

datasets_path = 'multilingual'  # to search a file or dir that is acceptable for 'dataset'
sent_classifier = Trainer(config=apc_config_multilingual,
                          dataset=datasets_path,
                          checkpoint_save_mode=1,
                          auto_device=True
                          ).load_trained_model()
