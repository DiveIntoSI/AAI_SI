##########################################################################################
# parameters

model_params = {
    'model_name': 'STNet',
}

optimizer_params = {
    'name': 'Adam',
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [201, ],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': True,
    'cuda_device_num': 0,
    'epochs': 100,
    'train_batch_size': 32,
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
    'n_spilt': 10,
    'add_noise': True,
    'feature_name': 'LogMel_Features'
}
##########################################################################################
# main
import logging
from utils.utils import create_logger, copy_all_src
#
from Trainer import Trainer


def main():
    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params,
                      dataset_params=dataset_params)
    copy_all_src(trainer.result_folder)
    trainer.run()



def _print_config():
    logger = logging.getLogger('root')
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(trainer_params['use_cuda'],
                                                           trainer_params['cuda_device_num']))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == '__main__':
    main()
