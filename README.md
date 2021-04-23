# undergraduate-thesis
这是本人的毕业论文代码，也是我的BraTS 2020挑战赛的解决方案。

# Training

首先先将图片剪裁成patch，运行`src/partition.ipy`，依次更改训练集，验证集，测试集的存储地址。

```python
train_brats_path = "your-Path_to/MICCAI_BraTS_2020_Data_Training"
output_trainImage = "your-Path_to/trainImage"
output_trainMask = "your-Path_to/trainMask"
```
然后更改`src/train.py`，更改data_path，开始运行

```python
data_path = 'your-Path_to'
```

# Inference

运行`src/inference.py`，将图片还原，

# Inference

运行`src/test.py`，进行测试。

