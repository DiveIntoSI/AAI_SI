
import torch
from logging import getLogger
from tqdm import tqdm
from torch import nn
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

from MyDataSet import MyDataSet
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self,model_params,optimizer_params,trainer_params, dataset_params):

        # save arguments
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.dataset_params = dataset_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        model_ = model_params["model"]
        self.model = model_(**self.model_params[model_params["model_name"]+"_params"])

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        # dataloader
        self.train_dataloader = None
        self.val_dataloader = None
        # loss
        self.loss = nn.CrossEntropyLoss()

        # 加载模型( 未验证
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        n_spilt = self.dataset_params['n_spilt']
        data_folder = self.dataset_params['data_folder']
        split_info_folder = self.dataset_params['split_info_folder']
        add_noise = self.dataset_params['add_noise']
        feature_name = self.dataset_params['feature_name']
        train_batch_size = self.trainer_params['train_batch_size']

        val_score_MX = 0.0
        val_score_MX_spilt_epoch = []

        for i_spilt in range(n_spilt):
            self.logger.info('=================================================================')
            self.logger.info('Begin Spilt {:3d}'.format(i_spilt))
            self.logger.info('=================================================================')
            train_info_txt = os.path.join(split_info_folder,
                                          'data_spilt' + f'_{n_spilt}',
                                          f'train_info{i_spilt}.txt')
            val_info_txt = os.path.join(split_info_folder,
                                        'data_spilt' + f'_{n_spilt}',
                                        f'val_info{i_spilt}.txt')
            train_dataset = MyDataSet(train_info_txt, add_noise, data_folder, feature_name)
            val_dataset = MyDataSet(val_info_txt, add_noise, data_folder, feature_name)
            self.train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size)
            self.val_dataloader = DataLoader(val_dataset, batch_size=train_batch_size)
            model = self.model_params["model"]
            self.model = model(**self.model_params[self.model_params["model_name"] + "_params"])
            val_score_MX, val_score_MX_spilt_epoch = self._run_i_spilt(i_spilt,
                                                                       val_score_MX,
                                                                       val_score_MX_spilt_epoch)
        # 加载模型并获得测试的结果，代写

    def _run_i_spilt(self, i_spilt, val_score_MX, val_score_MX_spilt_epoch):
        self.time_estimator.reset(self.start_epoch)
        for epoch in tqdm(range(self.start_epoch, self.trainer_params['epochs']+1)):
            self.logger.info('=================================================================')

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            # LR Decay
            self.scheduler.step()

            # 测验证集val
            val_score, val_loss = self._val_one_epoch(epoch)
            # save model
            if val_score > val_score_MX:
                self.logger.info(f"Best Val Scroe at Spilt {i_spilt} Epoch {epoch} Val Scroe{val_score}")
                val_score_MX = val_score
                val_score_MX_spilt_epoch = [i_spilt, epoch]
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, f'{self.result_folder}/checkpoint-BS-{i_spilt}_{epoch}.pt')

            # update val
            self.result_log.append('val_score', epoch, val_score)
            self.result_log.append('val_loss', epoch, val_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['val_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['val_loss'])

            if epoch % model_save_interval == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if epoch % img_save_interval == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['val_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['val_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
        return val_score_MX, val_score_MX_spilt_epoch

    def _train_one_epoch(self, epoch):
        batch_size = self.trainer_params['train_batch_size']

        # train
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        self.model.train()

        for loop_cnt, (datas, labels) in enumerate(self.train_dataloader):
            outputs = self.model(datas)
            loss = self.loss(outputs, labels.to(torch.int64))
            # back
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
            # acc
                predict = torch.max(nn.Softmax(dim=1)(outputs), dim=1).indices
                score = torch.mean((predict == labels).float())

            # update AM
            score_AM.update(score.item(), datas.shape[0])
            loss_AM.update(loss.item(), datas.shape[0])

            if epoch == 1 and loop_cnt <= 10:
                self.logger.info('Epoch {:3d}: Train {:3d},  Score: {:.4f},  Loss: {:.4f}'
                                 .format(epoch, loop_cnt, score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train,  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg


    def _val_one_epoch(self, epoch):
        batch_size = self.trainer_params['train_batch_size']

        # train
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        self.model.eval()

        for loop_cnt, (datas, labels) in enumerate(self.val_dataloader):
            outputs = self.model(datas)
            loss = self.loss(outputs, labels.to(torch.int64))

            # acc
            predict = torch.max(nn.Softmax(dim=1)(outputs), dim=1).indices
            score = torch.mean((predict == labels).float())

            # update AM
            score_AM.update(score.item(), batch_size)
            loss_AM.update(loss.item(), batch_size)

        # Log Once, for each epoch
        self.logger.info('Val Epoch {:3d}: Val,  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg