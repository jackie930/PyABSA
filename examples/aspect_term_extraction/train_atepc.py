# -*- coding: utf-8 -*-
# file: train_atepc.py
# time: 2021/5/21 0021
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.


########################################################################################################################
#                                               ATEPC training_tutorials script                                        #
########################################################################################################################
from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCConfigManager

atepc_config_english = ATEPCConfigManager.get_atepc_config_english()
atepc_config_english.num_epoch = 10
atepc_config_english.evaluate_begin = 4
atepc_config_english.lot_step = 100
atepc_config_english.model = ATEPCModelList.LCF_ATEPC
# SemEval = ABSADatasetList.SemEval
dataset_path = r'integrated_datasets/datasets/atepc_datasets/TShirt'
aspect_extractor = ATEPCTrainer(config=atepc_config_english,
                                dataset=dataset_path,
                                from_checkpoint='lcf_atepc_cdw_apcacc_80.7_apcf1_68.67_atef1_80.55',
                                checkpoint_save_mode=1,
                                auto_device=True
                                )
