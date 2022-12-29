import os
import random
import warnings
import librosa
import glob
import pickle
import torch
import fairseq
import argparse
import webrtcvad
import collections
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from python_speech_features import logfbank
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


class NoisePerturbAugmentor(object):
    """用于添加背景噪声的增强模型

    :param min_snr_dB: 最小的信噪比，以分贝为单位
    :type min_snr_dB: int
    :param max_snr_dB: 最大的信噪比，以分贝为单位
    :type max_snr_dB: int
    :param noise_path: 噪声文件夹
    :type noise_path: str
    :param sr: 音频采样率，必须跟训练数据的一样
    :type sr: int
    """

    def __init__(self, min_snr_dB=14, max_snr_dB=16, noise_path="data/noise", sr=16000):
        self.sr = sr
        self._min_snr_dB = min_snr_dB
        self._max_snr_dB = max_snr_dB
        self._noise_files = self.get_noise_file(noise_path=noise_path)

    # 获取全部噪声数据
    @staticmethod
    def get_noise_file(noise_path):
        noise_files = []
        if not os.path.exists(noise_path): return noise_files
        for file in os.listdir(noise_path):
            noise_files.append(os.path.join(noise_path, file))
        return noise_files

    @staticmethod
    def rms_db(wav):
        """返回以分贝为单位的音频均方能量

        :return: 均方能量(分贝)
        :rtype: float
        """
        mean_square = np.mean(wav ** 2)
        return 10 * np.log10(mean_square)

    def __call__(self, wav):
        """添加背景噪音音频

        :param wav: librosa 读取的数据
        :type wav: ndarray
        """
        # 如果没有噪声数据跳过
        if len(self._noise_files) == 0:
            print('could not find any noise file')
            exit(1)
        noise, _sr = librosa.load(random.choice(self._noise_files), sr=self.sr)
        # 噪声大小
        snr_dB = random.uniform(self._min_snr_dB, self._max_snr_dB)
        noise_gain_db = min(self.rms_db(wav) - self.rms_db(noise) - snr_dB, 300)
        noise *= 10. ** (noise_gain_db / 20.)
        # 合并噪声数据
        noise_new = np.zeros(wav.shape, dtype=np.float32)
        if noise.shape[0] >= wav.shape[0]:
            start = random.randint(0, noise.shape[0] - wav.shape[0])
            noise_new[:wav.shape[0]] = noise[start: start + wav.shape[0]]
        else:
            start = random.randint(0, wav.shape[0] - noise.shape[0])
            noise_new[start:start + noise.shape[0]] = noise[:]
        return wav + noise_new


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class Preprocess():
    def __init__(self, hparams):
        self.hparams = hparams
        self.silence_clip_num = list()
        self.silence_clip_ratio = list()
        self.audio_duration = list()
        self.NoiseAug = NoisePerturbAugmentor()
        cp_path = 'ckpt/wav2vec_small.pt'
        self.wav2vec_device = 'cuda'
        self.wav2vec_model, _cfg, _task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.wav2vec_model = self.wav2vec_model[0].to(self.wav2vec_device)
        self.wav2vec_model.eval()
        self.split_seq_num = 0

    def split_sample(self, sample_ary, seq_len, overlap_rate=0):
        sample_len = sample_ary.shape[0]
        split_seqs = list()
        seq_start = 0
        while True:
            seq_end = min(seq_start + seq_len, sample_len)
            missing_len = seq_len - (seq_end - seq_start)
            if missing_len > 0:
                if missing_len / seq_len > 0.3 and seq_start != 0:
                    break  # 舍弃空白长的片段
                seq_arr = np.hstack((sample_ary[seq_start:seq_end], np.zeros(missing_len)))
            else:
                seq_arr = sample_ary[seq_start:seq_end]
            split_seqs.append(seq_arr)
            self.split_seq_num += 1
            seq_start = seq_start - int(seq_len * overlap_rate) + seq_len
        return split_seqs

    def preprocess_data(self):
        path_list = []
        if self.hparams.mode == 'train':
            path_list = [x for x in glob.iglob(os.path.join(self.hparams.in_dir.rstrip("/") + "/*/*.flac"))]
        elif self.hparams.mode == 'test':
            path_list = [x for x in glob.iglob(os.path.join(self.hparams.in_dir.rstrip("/") + "/*.flac"))]
        for path in tqdm(path_list):
            wav_arr, sample_rate = self.vad_process(path)
            # padding
            split_len = int(self.hparams.segment_length * sample_rate)
            # 训练集分割时不重叠，测试集重叠30%
            split_slices = self.split_sample(wav_arr, split_len,
                                             overlap_rate=0 if self.hparams.mode == 'train' else 0.3)
            for seq_index, seq_ary in enumerate(split_slices):
                self.create_pickle(path, seq_ary, sample_rate, seq_index=seq_index, noised=False)
                if self.hparams.mode == 'train':  # 对训练集保存加噪声处理结果
                    wav_arr_noised_ = self.NoiseAug((seq_ary / (2 ** 15)).astype(np.float32))  # 这里将int16格式”转“为float32
                    wav_arr_noised = (wav_arr_noised_ * (2 ** 15)).astype(np.int16)  # 再转回int16
                    # sf.write('tmp.wav', wav_arr_noised, 16000)  #保存修改后的声音，以试听。
                    self.create_pickle(path, wav_arr_noised, sample_rate, seq_index=seq_index, noised=True)
        plt.clf()
        plt.hist(self.silence_clip_ratio)
        plt.xlabel('silence ratio')
        plt.ylabel('sample num')
        plt.savefig(f'output/{self.hparams.mode}_silence_clip_ratio_histogram.png')
        plt.clf()
        plt.hist(self.silence_clip_num)
        plt.xlabel('silence clip num')
        plt.ylabel('sample num')
        plt.savefig(f'output/{self.hparams.mode}_silence_clip_num_histogram.png')
        plt.clf()
        plt.hist(self.audio_duration)
        plt.xlabel('audio duration/s')
        plt.ylabel('sample num')
        plt.savefig(f'output/{self.hparams.mode}_audio_duration_histogram.png')
        plt.close()
        print('total split seq num:', self.split_seq_num)

    def vad_process(self, path):
        # VAD Process
        audio_, sample_rate = sf.read(path, dtype='int16')
        audio = audio_.tobytes()  # 16bits=2bytes   audio_duration=len(audio)/2/sample_rate
        vad = webrtcvad.Vad(1)  # 识别空段的强度，0最弱，3最强。
        frames = self.frame_generator(30, audio, sample_rate)
        frames = list(frames)  # 输入9s音频的话， 将生成 9s/30ms=300帧。
        segments = self.vad_collector(sample_rate, 30, 300, vad, frames)
        total_wav = b""
        n_segments = 0
        for i, segment in enumerate(segments):
            total_wav += segment
            n_segments += 1
        self.silence_clip_num.append(n_segments - 1)  # 统计静音片段数 每个静音片段进行一次分割
        self.silence_clip_ratio.append(1 - len(total_wav) / len(audio))  # 统计静音比例
        self.audio_duration.append(len(audio) / 2 / sample_rate)  # 统计音频时长
        # Without writing, unpack total_wav into numpy [N,1] array
        # 16bit PCM 기준 dtype=np.int16
        wav_arr = np.frombuffer(total_wav, dtype=np.int16)
        # print("read audio data from byte string. np array of shape:"+str(wav_arr.shape))
        return wav_arr, sample_rate

    def frame_generator(self, frame_duration_ms, audio, sample_rate):
        """Generates audio frames from PCM audio data.
        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.
        Yields Frames of the requested duration.
        """
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 每次采样16位(2bytes)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(self, sample_rate, frame_duration_ms,
                      padding_duration_ms, vad, frames):
        """Filters out non-voiced audio frames.
        Given a webrtcvad.Vad and a source of audio frames, yields only
        the voiced audio.
        Uses a padded, sliding window algorithm over the audio frames.
        When more than 90% of the frames in the window are voiced (as
        reported by the VAD), the collector triggers and begins yielding
        audio frames. Then the collector waits until 90% of the frames in
        the window are unvoiced to detrigger.
        The window is padded at the front and back to provide a small
        amount of silence or the beginnings/endings of speech around the
        voiced frames.
        Arguments:
        sample_rate - The audio sample rate, in Hz.
        frame_duration_ms - The frame duration in milliseconds.
        padding_duration_ms - The amount to pad the window, in milliseconds.
        vad - An instance of webrtcvad.Vad.
        frames - a source of audio frames (sequence or generator).
        Returns: A generator that yields PCM audio data.
        """
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        # We use a deque for our sliding window/ring buffer.
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
        # NOTTRIGGERED state.
        triggered = False

        voiced_frames = []
        for frame in frames:
            is_speech = vad.is_speech(frame.bytes, sample_rate)

            # sys.stdout.write('1' if is_speech else '0')
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                # If we're NOTTRIGGERED and more than 60% of the frames in
                # the ring buffer are voiced frames, then enter the
                # TRIGGERED state.
                if num_voiced >= 0.6 * ring_buffer.maxlen:
                    triggered = True
                    # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                    # We want to yield all the audio we see from now until
                    # we are NOTTRIGGERED, but we have to start with the
                    # audio that's already in the ring buffer.
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                # We're in the TRIGGERED state, so collect the audio data
                # and add it to the ring buffer.
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                # If more than 90% of the frames in the ring buffer are
                # unvoiced, then enter NOTTRIGGERED and yield whatever
                # audio we've collected.
                if num_unvoiced >= 0.6 * ring_buffer.maxlen:
                    # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                    triggered = False
                    yield b''.join([f.bytes for f in voiced_frames])
                    ring_buffer.clear()
                    voiced_frames = []
        # if triggered:
        # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        # sys.stdout.write('\n')
        # If we have any leftover voiced audio when we run out of input,
        # yield it.
        if voiced_frames:
            yield b''.join([f.bytes for f in voiced_frames])

    def wav2vec(self, wav_arr):
        wav_arr = wav_arr.astype(np.int32)
        wav_arr_normed = (wav_arr - wav_arr.min()) / (wav_arr.max() - wav_arr.min())
        wav_arr_standardized = (wav_arr_normed - wav_arr_normed.mean()) / wav_arr_normed.std()
        inp = torch.tensor(wav_arr_standardized).to(self.wav2vec_device).float()
        output_feats = self.wav2vec_model(inp[None, :], mask=False, features_only=True)['features']
        return output_feats

    def create_pickle(self, path, wav_arr, sample_rate, seq_index, noised=False):
        # 目前仅提取了 logmel_feats 特征
        save_dict = {}
        logmel_feats = logfbank(
            wav_arr, samplerate=sample_rate, nfilt=self.hparams.spectrogram_scale)
        # print("created logmel feats from audio data. np array of shape:"+str(logmel_feats.shape))
        save_dict["LogMel_Features"] = logmel_feats
        wav2vec_feats = self.wav2vec(wav_arr)  # 下面开始使用预训练模型wav2vec生成特征
        save_dict["Wav2vec_Features"] = wav2vec_feats[0].detach().cpu().numpy()
        save_dict["Audio_Ary"] = wav_arr
        pickle_f_name = None
        if self.hparams.mode == 'train':
            pickle_f_name = path.split("/")[-1].replace(".flac", f"_{seq_index:03d}.pickle")
            if noised: pickle_f_name = pickle_f_name.replace('.pickle', '_noised.pickle')  # 对训练集保存加噪声处理结果
            save_dict["SpkId"] = path.split("/")[-2]
            save_dict["WavId"] = path.split("/")[-1].split(".")[-2].split("_")[-1]
        elif self.hparams.mode == 'test':
            pickle_f_name = path.split("/")[-1].replace(".flac", f"_{seq_index:03d}.pickle")
            save_dict["WavId"] = path.split("/")[-1].split(".")[-2][4:]

        if not os.path.exists(self.hparams.pk_dir):
            os.mkdir(self.hparams.pk_dir)
        # print(os.path.join(self.hparams.pk_dir.rstrip("/"), pickle_f_name))
        with open(os.path.join(self.hparams.pk_dir.rstrip("/"), pickle_f_name), "wb") as f:
            pickle.dump(save_dict, f, protocol=3)


def main():
    # Hyperparameters
    parser = argparse.ArgumentParser()

    # in_dir
    parser.add_argument("--in_dir", type=str, default='data/train',
                        help="input audio data dir")
    parser.add_argument("--pk_dir", type=str, default='data_pk/train',
                        help="output pickle dir")
    parser.add_argument("--mode", default='train',
                        choices=["train", "test"])

    # Data Process
    parser.add_argument("--segment_length", type=float,
                        default=3.015, help="segment length in seconds")
    parser.add_argument("--spectrogram_scale", type=int, default=40,
                        help="scale of the input spectrogram")
    args = parser.parse_args()

    preprocess = Preprocess(args)
    preprocess.preprocess_data()


if __name__ == "__main__":
    print('ok')
    main()
