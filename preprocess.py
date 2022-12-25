import os
import glob
import pickle
import argparse
import webrtcvad
import collections
import numpy as np
import pandas as pd
import soundfile as sf
from python_speech_features import logfbank
from sklearn.model_selection import StratifiedShuffleSplit




class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class Preprocess():
    def __init__(self, hparams):
        self.hparams = hparams

    def preprocess_data(self):
        path_list = []
        if self.hparams.mode == 'train':
            path_list = [x for x in glob.iglob(os.path.join(self.hparams.in_dir.rstrip("/") + "/*/*.flac"))]
        elif self.hparams.mode == 'test':
            path_list = [x for x in glob.iglob(os.path.join(self.hparams.in_dir.rstrip("/") + "/*.flac"))]
        for path in path_list:
            wav_arr, sample_rate = self.vad_process(path)
            # 在这里加一个保存文件  待写
            # padding
            singal_len = int(self.hparams.segment_length * sample_rate)
            n_sample = wav_arr.shape[0]
            if n_sample < singal_len:
                wav_arr = np.hstack((wav_arr, np.zeros(singal_len - n_sample)))
            else:
                wav_arr = wav_arr[(n_sample - singal_len) //
                                  2:(n_sample + singal_len) // 2]
            # self.create_pickle(path, wav_arr, sample_rate)

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
        # print(f'seg num:{n_segments:2d} {len(total_wav) / len(audio):.3f} {len(audio)/2/sample_rate:.1f}s {path}')
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

    def create_pickle(self, path, wav_arr, sample_rate):
        # 目前仅提取了 logmel_feats 特征
        if round((wav_arr.shape[0] / sample_rate), 1) >= self.hparams.segment_length:
            save_dict = {}
            logmel_feats = logfbank(
                wav_arr, samplerate=sample_rate, nfilt=self.hparams.spectrogram_scale)
            # print("created logmel feats from audio data. np array of shape:"+str(logmel_feats.shape))

            save_dict["LogMel_Features"] = logmel_feats
            pickle_f_name = None
            if self.hparams.mode == 'train':
                pickle_f_name = path.split("/")[-1].replace("flac", "pickle")
                save_dict["SpkId"] = path.split("/")[-2]
                save_dict["WavId"] = path.split("/")[-1].split(".")[-2].split("_")[-1]
            elif self.hparams.mode == 'test':
                pickle_f_name = path.split("/")[-1].replace("flac", "pickle")
                save_dict["WavId"] = path.split("/")[-1].split(".")[-2][4:]

            if not os.path.exists(self.hparams.pk_dir):
                os.mkdir(self.hparams.pk_dir)
            print(os.path.join(self.hparams.pk_dir.rstrip("/"), pickle_f_name))
            with open(os.path.join(self.hparams.pk_dir.rstrip("/"), pickle_f_name), "wb") as f:
                pickle.dump(save_dict, f, protocol=3)
        else:
            print("wav length smaller than 1.6s: " + path)  # 按照目前的超参数设置(3秒)，这段代码没用，不会有小于3秒的


def save_train_val_txt(train_folder, data_folder):
    # 对train_folder数据集进行分层划分,txt存入data_folder下
    #  train_floder: 'data/train'
    data = []
    for speaker in os.listdir(train_folder):
        label = int(speaker[-3:])
        for speaker_one in os.listdir(os.path.join(train_folder, speaker)):
            FileID = os.path.join(train_folder, speaker, speaker_one)
            data.append([FileID, label])

    df = pd.DataFrame(data, columns=['FileID', 'Label'])

    # 对train按照比例划分出训练集和验证集
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=111)

    train_index, test_index = list(split.split(df, df["Label"]))[0]

    df.loc[train_index.tolist()].to_csv(os.path.join(data_folder, "train_info.txt"), index=False)
    df.loc[test_index.tolist()].to_csv(os.path.join(data_folder, "val_info.txt"), index=False)



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
                        default=3, help="segment length in seconds")
    parser.add_argument("--spectrogram_scale", type=int, default=40,
                        help="scale of the input spectrogram")
    args = parser.parse_args()

    preprocess = Preprocess(args)
    preprocess.preprocess_data()


if __name__ == "__main__":
    print('ok')
    main()
