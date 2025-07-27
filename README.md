## Generate Code

~~~bash
生成核心运行部分在\Experiments\chb_multiple_classes.ipynb或者demo.py中
~~~

## Generate Data or Train Data

~~~bash
生成、测试以及训练数据文件过大，需要的可以联系作者
~~~

## Metric Code

~~~bash
评估核心运行在\Experiments\metric_pytorch.ipynb和Experiments\metric_tensorflow.ipynb中

评估部分的数据替换：
示例：
将\Data\aug_train\full-0.2-200-raw.npy替换ori_data = np.load('../chb_exp/ori_data_1.npy')中的数据路径
将\Data\aug_train\full-0.2-200.npy替换fake_data = np.load('../chb_exp/ddpm_fake_1_eeg.npy')中的数据路径
~~~

## Classic Code

~~~bash
分类核心运行\Experiments\model.ipynb
分类部分的数据替换同上
~~~
