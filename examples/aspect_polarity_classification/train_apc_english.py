# -*- coding: utf-8 -*-
# file: train_apc_english.py
# time: 2021/6/8 0008
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.

########################################################################################################################
#                  train and evaluate on your own atepc_datasets (need train and test atepc_datasets)                  #
#              your custom dataset should have the continue polarity labels like [0,N-1] for N categories              #
########################################################################################################################

from pyabsa import train_apc, get_apc_param_dict_english

from pyabsa.dataset import laptop14, restaurant14, restaurant15, restaurant16
apc_param_dict_english = get_apc_param_dict_english()

# apc_param_dict_english['model_name'] = 'slide_lcfs_bert'
# apc_param_dict_english['lcf'] = 'cdw'
# apc_param_dict_english['log_step'] = 3
# apc_param_dict_english['l2reg'] = 0.00005
# apc_param_dict_english['seed'] = {1, 2, 3}
# apc_param_dict_english['evaluate_begin'] = 2


save_path = 'state_dict'

# sent_classifier = train_apc(parameter_dict=apc_param_dict_english, # set param_dict=None will use the apc_param_dict as well
#                             dataset_path=laptop14,    # train set and test set will be automatically detected
#                             model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
#                             auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
#                             auto_device=True               # automatic choose CUDA or CPU
#                             )

sent_classifier = train_apc(parameter_dict=apc_param_dict_english,     # set param_dict=None to use default model
                            dataset_path=restaurant14,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

sent_classifier = train_apc(parameter_dict=apc_param_dict_english,     # set param_dict=None to use default model
                            dataset_path=restaurant15,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

sent_classifier = train_apc(parameter_dict=apc_param_dict_english,     # set param_dict=None to use default model
                            dataset_path=restaurant16,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

apc_param_dict_english['lcf'] = 'cdm'

sent_classifier = train_apc(parameter_dict=apc_param_dict_english, # set param_dict=None will use the apc_param_dict as well
                            dataset_path=laptop14,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

sent_classifier = train_apc(parameter_dict=apc_param_dict_english,     # set param_dict=None to use default model
                            dataset_path=restaurant14,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

sent_classifier = train_apc(parameter_dict=apc_param_dict_english,     # set param_dict=None to use default model
                            dataset_path=restaurant15,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )

sent_classifier = train_apc(parameter_dict=apc_param_dict_english,     # set param_dict=None to use default model
                            dataset_path=restaurant16,    # train set and test set will be automatically detected
                            model_path_to_save=save_path,  # set model_path_to_save=None to avoid save model
                            auto_evaluate=True,            # evaluate model while training_tutorials if test set is available
                            auto_device=True               # automatic choose CUDA or CPU
                            )


# custom_param_dict = {'model_name': 'slide_lcfs_bert',    # {slide_lcfs_bert, slide_lcf_bert, lcfs_bert, lcf_bert, bert_spc, bert_base, lca_bert}
#               'batch_size': 16,
#               'seed': {1, 2, 3},           # you can use a set of random seeds to train multiple rounds
#               # 'seed': 996,               # or use one seed only
#               'num_epoch': 10,
#               'optimizer': "adam",         # {adam, adamw}
#               'learning_rate': 0.00002,
#               'pretrained_bert_name': "bert-base-uncased",
#               'use_dual_bert': False,      # modeling the local and global context using different BERTs
#               'use_bert_spc': True,        # Enable to enhance APC, do not use this parameter in ATE
#               'max_seq_len': 80,
#               'log_step': 3,               # Evaluate per steps
#               'SRD': 3,                    # Distance threshold to calculate local context
#               # 'use_syntax_based_SRD': True,   # force to use syntax-based semantic-relative distance in all lcf-based models
#               'eta': -1,                   # Eta is valid in [0,1] slide_lcf_bert/slide_lcf_bert
#               'sigma': 0.3,                # Sigma is valid in LCA-Net, ranging in [0,1]
#               'lcf': "cdw",                # {cdm, cdw} valid in lcf-bert models
#               'window': "lr",              # {lr, l, r} valid in slide_lcf_bert/slide_lcf_bert
#               'dropout': 0,
#               'l2reg': 0.00001,
#               # 'polarities_dim': 2,       # deprecated, polarities_dim will be automatic detected
#               'dynamic_truncate': True,    # Dynamic truncate the text according to the position of aspect term
#               'evaluate_begin': 0  # evaluate begin with epoch
#               }
