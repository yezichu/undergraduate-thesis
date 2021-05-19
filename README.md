# undergraduate-thesis
This is the code for my graduation thesis and the solution for my BRATS 2020 Challenge.

# Image processing
(/image/data_sample.png)
Here are the four modes of the image and its mask

# Training

First, the image is cropped into patches，`srrunc/partition.ipy`，Change the storage address of training set, verification set and test set successively.

```python
train_brats_path = "your-Path_to/MICCAI_BraTS_2020_Data_Training"
output_trainImage = "your-Path_to/trainImage"
output_trainMask = "your-Path_to/trainMask"
```
And change the`src/train.py`，change data_path.Began to run

```python
data_path = 'your-Path_to'
```

# Inference

Run`src/inference.py`，to restore the picture.

# Test

Run`src/test.py`.Test it.

# Experimental results

 ![Train](/image/对比图.png)
![Train](/image/对比图2.png)
![Test](/image/验证集对比图.png)



