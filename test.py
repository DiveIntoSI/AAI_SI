import os
import pickle
from tqdm import tqdm
import torch
from Model.MLPModel import MLPModel
from Model.SAEPModel import SAEPModel

model_params = {
    'model': SAEPModel,
    'model_name': "SAEPModel",
    'SAEPModel_params': {
        'seq_len': 300,
        'input_dim': 40,
        'hidden_dim': 512,
        'dense_dim': (128, 250, 250)
    },
}
test_params = {
    'use_cuda': True,
    'cuda_device_num': 4,
}
dataset_params = {
    'test_data_folder': 'data_pk/test-noisy',
    'feature_name': 'LogMel_Features',  # LogMel_Features Wav2vec_Features
    'predict_results_path': 'results_test_noisy.txt',
}

if __name__ == '__main__':
    checkpoint_fullname = r'/data2/haoxiaoyang/prj/d2si/AAI_SI/result/20221230_184247_sid/checkpoint-BS-0_100.pt'
    USE_CUDA = test_params['use_cuda']
    if USE_CUDA:
        cuda_device_num = test_params['cuda_device_num']
        device = torch.device('cuda', cuda_device_num)
    else:
        device = torch.device('cpu')
    test_data_folder = dataset_params['test_data_folder']
    feature_name = dataset_params['feature_name']
    predict_results_path = dataset_params['predict_results_path']
    model_ = model_params["model"]
    model = model_(**model_params[model_params["model_name"] + "_params"]).to(device)
    checkpoint = torch.load(checkpoint_fullname, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # '.flac : (250)概率'
    sample_dict = dict()

    all_slice = os.listdir(test_data_folder)
    for one in tqdm(all_slice):
        sample = one[:8] + '.flac'
        if sample not in sample_dict:
            sample_dict[sample] = torch.zeros(250)

        item = os.path.join(test_data_folder, one)
        with open(item, "rb") as f:
            load_dict = pickle.load(f)
            data = load_dict[feature_name]
            data = torch.Tensor(data).unsqueeze(0)  # batch_size1
        with torch.no_grad():
            outputs = model(data.to(device)).squeeze().cpu()
        sample_dict[sample] += outputs

    # 遍历speaker_dict，求一次max然后保存
    if os.path.exists(predict_results_path):
        os.remove(predict_results_path)
    file_write_obj = open(predict_results_path, 'w')
    for key, value in sample_dict.items():
        predict = torch.argmax(value)
        predict += 1
        file_write_obj.writelines(f'{key} spk{predict:0>3d}\n')
    file_write_obj.close()
