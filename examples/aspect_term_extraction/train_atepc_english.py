# -*- coding: utf-8 -*-
# file: train_atepc_english.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                                               ATEPC training_tutorials script                                        #
########################################################################################################################


from pyabsa import train_atepc, get_atepc_param_dict_english

from pyabsa import ABSADatasets

# see hyper-parameters in pyabsa/main/training_configs.py
param_dict = {'model_name': 'bert_base',
              'batch_size': 16,
              'seed': {996},
              'num_epoch': 6,
              'optimizer': "adam",    # {adam, adamw}
              'learning_rate': 0.00003,
              'pretrained_bert_name': "bert-base-uncased",
              'use_dual_bert': False,  # modeling the local and global context using different BERTs
              'use_bert_spc': False,   # enable to enhance APC, not available for ATE or joint module of APC and ATE
              'max_seq_len': 80,
              'log_step': 50,           # evaluate per steps
              'SRD': 3,                # distance threshold to calculate local context
              'use_syntax_based_SRD': True,   # force to use syntax-based semantic-relative distance in all lcf-based models
              'lcf': "cdw",            # {cdw, cdm, fusion}
              'dropout': 0.5,
              'l2reg': 0.00001,
              'evaluate_begin': 5  # evaluate begin with epoch
              # 'polarities_dim': 3      # deprecated, polarity_dim will be automatically detected
              }

save_path = 'state_dict'

# param_dict = get_atepc_param_dict_english()
semeval = ABSADatasets.semeval
aspect_extractor = train_atepc(parameter_dict=param_dict,      # set param_dict=None to use default model
                               dataset_path=semeval,    # file or dir, dataset(s) will be automatically detected
                               model_path_to_save=save_path,   # set model_path_to_save=None to avoid save model
                               auto_evaluate=True,             # evaluate model while training_tutorials if test set is available
                               auto_device=True                # Auto choose CUDA or CPU
                               )
