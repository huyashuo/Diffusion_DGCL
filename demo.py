import os
import sys
sys.path.append(os.path.join(os.path.dirname('__file__'), '../'))

import torch
import numpy as np

from engine.solver import Trainer
from Utils.metric_utils import visualization
from Data.build_dataloader import build_dataloader
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one, cond_fn
from Utils.marginal_calibration_eeg import marginal_calibration_eeg
import matplotlib.pyplot as plt

print(f'cuda' if torch.cuda.is_available() else 'cpu')

print(torch.__version__)  # 输出 PyTorch 版本（如 2.0.1）
print(torch.version.cuda) # 查看 PyTorch 关联的 CUDA 版本（如果有 GPU）
print(torch.cuda.is_available())  # 检查 GPU 是否可用（输出 True 或 False）


class Args_Example:
    def __init__(self) -> None:
        self.config_path = '/mnt/proj/Diffusion_DGCL/Config/chb.yaml'
        self.gpu = 0
        self.save_dir = '/mnt/proj/Diffusion_DGCL/chb_exp'
        os.makedirs(self.save_dir, exist_ok=True)

args =  Args_Example()
configs = load_yaml_config(args.config_path)
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

dl_info = build_dataloader(configs, args)
model = instantiate_from_config(configs['model']).to(device)
classifier = instantiate_from_config(configs['classifier']).to(device)
trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)

trainer.train()

trainer.train_classfier(classifier)

dataset = dl_info['dataset']
seq_length, feature_dim = dataset.window, dataset.var_num

model_kwargs = {}
model_kwargs['classifier'] = trainer.classifier
model_kwargs['classifier_scale'] = 0.1

ori_data_1 = dataset.normalize(dataset.data_1)

model_kwargs['y'] = torch.ones((1001, )).long().to(device)

fake_data_1 = trainer.sample(num=len(ori_data_1), size_every=1001, shape=[seq_length, feature_dim],
                             model_kwargs=model_kwargs, cond_fn=cond_fn)
fake_data_2 = trainer.sample(num=(len(ori_data_1)+3000), size_every=1001, shape=[seq_length, feature_dim],
                             model_kwargs=model_kwargs, cond_fn=cond_fn)
fake_data_3 = trainer.sample(num=(len(ori_data_1)+5000), size_every=1001, shape=[seq_length, feature_dim],
                             model_kwargs=model_kwargs, cond_fn=cond_fn)

if dataset.auto_norm:
    ori_data_1 = unnormalize_to_zero_to_one(ori_data_1)
    fake_data_1 = unnormalize_to_zero_to_one(fake_data_1)

calibrated = marginal_calibration_eeg(fake_data_1, ori_data_1)
calibrated2 = marginal_calibration_eeg(fake_data_2, ori_data_1)
calibrated3 = marginal_calibration_eeg(fake_data_3, ori_data_1)

np.save(os.path.join(args.save_dir, f'calibrated_eeg.npy'), calibrated)
np.save(os.path.join(args.save_dir, f'ddpm_fake_1_eeg.npy'), fake_data_1)
np.save(os.path.join(args.save_dir, f'ori_data_1.npy'), ori_data_1)

np.save(os.path.join(args.save_dir, f'calibrated_eeg2.npy'), calibrated2)
np.save(os.path.join(args.save_dir, f'ddpm_fake_2_eeg.npy'), fake_data_2)
np.save(os.path.join(args.save_dir, f'calibrated_eeg3.npy'), calibrated3)
np.save(os.path.join(args.save_dir, f'ddpm_fake_3_eeg.npy'), fake_data_3)

# 任意挑一个时间步和通道，例如 t=64, c=2
plt.hist(fake_data_1[:, 24, 14], bins=50, alpha=0.4, label='Before')
plt.hist(calibrated[:, 24, 14], bins=50, alpha=0.4, label='After')
plt.hist(ori_data_1[:, 24, 14], bins=50, alpha=0.4, label='Real')
plt.legend()
plt.title("Marginal Calibration @ time=64, channel=2")
plt.show()

visualization(ori_data=ori_data_1, generated_data=fake_data_1, analysis='pca', compare=ori_data_1.shape[0])

visualization(ori_data=ori_data_1, generated_data=calibrated, analysis='pca', compare=ori_data_1.shape[0])

visualization(ori_data=ori_data_1, generated_data=fake_data_1, analysis='tsne', compare=ori_data_1.shape[0])

visualization(ori_data=ori_data_1, generated_data=calibrated, analysis='tsne', compare=ori_data_1.shape[0])

visualization(ori_data=ori_data_1, generated_data=fake_data_1, analysis='kernel', compare=ori_data_1.shape[0])

visualization(ori_data=ori_data_1, generated_data=calibrated, analysis='kernel', compare=ori_data_1.shape[0])