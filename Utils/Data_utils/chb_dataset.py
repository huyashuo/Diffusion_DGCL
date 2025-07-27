import os
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from sklearn.preprocessing import MinMaxScaler


class CHBDataset(Dataset):
    def __init__(
            self,
            data_root,
            window=256,
            save2npy=True,
            neg_one_to_one=True,
            period='train',
            output_dir='./OUTPUT',
            rate=0.5
    ):
        super(CHBDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'

        self.auto_norm, self.save2npy = neg_one_to_one, save2npy
        self.data_0, self.data_1, self.scaler = self.read_data(data_root, window)
        print(data_root)

        # 按比例划分训练集和测试集 (80% 训练, 20% 测试)
        X_train_0, X_test_0 = train_test_split(self.data_0, test_size=0.2, random_state=42)
        X_train_1, X_test_1 = train_test_split(self.data_1, test_size=0.2, random_state=42)
        y_train_0 = np.zeros(X_train_0.shape[0]).astype(np.int64)
        y_test_0 = np.zeros(X_test_0.shape[0]).astype(np.int64)
        y_train_1 = np.ones(X_train_1.shape[0]).astype(np.int64)
        y_test_1 = np.ones(X_test_1.shape[0]).astype(np.int64)


        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        raw_data_train = np.vstack([X_train_0, X_train_1])
        raw_label_train = np.concatenate([y_train_0, y_train_1])
        np.savez_compressed(os.path.join(self.dir, 'train'), data=raw_data_train, target=raw_label_train)

        raw_data_test = np.vstack([X_test_0, X_test_1])
        raw_label_test = np.concatenate([y_test_0, y_test_1])
        np.savez_compressed(os.path.join(self.dir, 'test'), data=raw_data_test, target=raw_label_test)

        print(f'提取前的X_train_1={X_train_1.shape}')
        # 计算需要提取的样本数
        np.random.shuffle(X_train_1)
        num_samples_to_extract = int(len(X_train_1) * rate)
        self.data_1 = X_train_1[:num_samples_to_extract]
        self.data_0 = X_train_0
        print(f'提取后的data1={self.data_1.shape}')
        self.labels_0 = np.zeros(self.data_0.shape[0]).astype(np.int64)
        self.labels_1 = np.ones(self.data_1.shape[0]).astype(np.int64)
        self.labels = np.concatenate([self.labels_0, self.labels_1])
        self.rawdata = np.vstack([self.data_0, self.data_1])

        np.savez_compressed(os.path.join(self.dir, 'train_1'), data=self.data_1, target=self.labels_1)

        np.savez_compressed(os.path.join(self.dir, f'train-{rate}'), data=self.rawdata, target=self.labels)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]

        self.samples = self.normalize(self.rawdata)

        self.sample_num = self.samples.shape[0]

    def read_data(self, filepath, length):
        """
        Reads the data from the given filepath, removes outliers, classifies the data into two classes,
        and scales the data using MinMaxScaler.

        Args:
            filepath (str): Path to the .arff file containing the EEG data.
            length (int): Length of the window for classification.
        """
        train_data = np.load(filepath)
        X_train = train_data['train_signals']
        X_train = np.transpose(X_train, (0, 2, 1))
        y_train = train_data['train_labels']
        print(X_train.shape)
        print(y_train.shape)

        # 按类别分割数据
        data_0 = X_train[y_train == 0]  # 非癫痫发作信号
        data_1 = X_train[y_train == 1]  # 癫痫发作信号
        print(f'X_non_seizure-->{data_0.shape}')
        print(f'X_seizure-->{data_1.shape}')

        data = np.vstack([data_0.reshape(-1, data_0.shape[-1]), data_1.reshape(-1, data_1.shape[-1])])
        print(f'data-->{data.shape}')

        scaler = MinMaxScaler()
        scaler = scaler.fit(data)

        return data_0, data_1, scaler

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            y = self.labels[ind]  # (1,) int
            return torch.from_numpy(x).float(), torch.tensor(y)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num

    def normalize(self, sq):
        d = self.__normalize(sq.reshape(-1, self.var_num))
        data = d.reshape(-1, self.window, self.var_num)
        return data

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)

    def shift_period(self, period):
        assert period in ['train', 'test'], 'period must be train or test.'
        self.period = period


def main():
    from torch.utils.data import DataLoader
    train_dataset = CHBDataset(data_root="D:\dev-project\Diffusion_DGCL\Data\datasets\EEG_Eye_State.arff",
                               window=24, save2npy=True, neg_one_to_one=True, period='train', output_dir='./OUTPUT')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


if __name__ == '__main__':
    main()