##########################################################################################
# parameters


from Model.MLPModel import MLPModel
from Model.SAEPModel import SAEPModel

model_params = {
    'model': SAEPModel,
    'model_name': "SAEPModel",
    'MLPModel_params': {
        'input_dim': 300*40,
        'ff_hidden_dim': 256,
        'output_dim': 250
    },
    'SAEPModel_params': {
        'seq_len': 300,
        'input_dim': 40,
        'hidden_dim': 512,
        'dense_dim': (128,250,250)
    },
}

optimizer_params = {
    'name': 'Adam',
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [90, ],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': True,
    'cuda_device_num': 0,
    'epochs': 100,
    'train_batch_size': 512,
    "model_load": {
        "enable": False,
        "path": str,
        "epoch": int
    },
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_sid.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    }
}

logger_params = {
    'log_file': {
        'desc': 'sid',
        'filename': 'run_log'
    }
}

dataset_params = {
    'data_folder': 'data_pk/train',
    'split_info_folder': 'data_spilt',
    'n_spilt': 3,
    'add_noise': True,
    'feature_name': 'LogMel_Features'
}
##########################################################################################
# main
import logging
from utils.utils import create_logger, copy_all_src
#
from Trainer import Trainer
from MyDataSet import save_train_val_txt

def main():
    create_logger(**logger_params)
    _print_config()
    save_train_val_txt(dataset_params)
    trainer = Trainer(model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params,
                      dataset_params=dataset_params)
    # copy_all_src(trainer.result_folder)
    trainer.run()



def _print_config():
    logger = logging.getLogger('root')
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(trainer_params['use_cuda'],
                                                           trainer_params['cuda_device_num']))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == '__main__':
    main()
