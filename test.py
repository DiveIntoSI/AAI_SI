import os
import pickle

import torch






def test(i_spilt, epoch):
    USE_CUDA = trainer_params['use_cuda']
    if USE_CUDA:
        cuda_device_num = trainer_params['cuda_device_num']
        device = torch.device('cuda', cuda_device_num)
    else:
        device = torch.device('cpu')
    test_data_folder = dataset_params['test_data_folder']
    feature_name = dataset_params['feature_name']
    predict_results_path = dataset_params['predict_results_path']

    checkpoint_fullname = f'{result_folder}/checkpoint-BS-{i_spilt}_{epoch}.pt'
    checkpoint = torch.load(checkpoint_fullname, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # '.flac : (250)概率'
    speaker_dict = dict()

    for one in os.listdir(test_data_folder):
        speaker = one[:8] + '.flac'
        if speaker not in speaker_dict:
            speaker_dict[speaker] = torch.zeros(250)

        item = os.path.join(test_data_folder, one)
        with open(item, "rb") as f:
            load_dict = pickle.load(f)
            data = load_dict[feature_name]
            data = torch.Tensor(data).unsqueeze(0) # batch_size1
        outputs = model(data).squeeze()
        speaker_dict[speaker] += outputs

    # 遍历speaker_dict，求一次max然后保存
    for key, value in speaker_dict:
        predict = torch.max(value).indices
        predict += 1
        speaker_dict[key] = f'spk{predict:0>3d}'

