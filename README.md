# undergraduate-thesis
This is the code for my graduation thesis and the solution for my BRATS 2020 Challenge.

# Image processing
![image](/image/data_sample.png)
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

**训练集对比**
![Train](/image/对比图.png)
**训练集对比**
![Train](/image/对比图2.png)
**验证集对比**
![Test](/image/验证集对比图.png)

![1](http://latex.codecogs.com/svg.latex?\begin{tabular}{ccccc}
\hline & ET & WT & TC & Mean \\
\hline Dice & $0.79655$ & $0.93$ & $0.90825$ & $0.878267$ \\
Sensitivity & $0.78583$ & $0.91073$ & $0.91593$ & $0.87083$ \\
Specificity & $0.99978$ & $0.99959$ & $0.9996$ & $0.999657$ \\
Hausdorff95 & $22.91558$ & $3.84997$ & $3.74955$ & $10.1717$ \\
\hline
\end{tabular})

![2](http://latex.codecogs.com/svg.latex?\begin{tabular}{ccccc}
\hline & ET & WT & TC & Mean \\
\hline Dice & $0.71534$ & $0.89413$ & $0.80496$ & $0.80481$ \\
Sensitivity & $0.70984$ & $0.88827$ & $0.81775$ & $0.805287$ \\
Specificity & $0.99971$ & $0.99917$ & $0.99938$ & $0.99942$ \\
Hausdorff95 & $36.16794$ & $4.63229$ & $9.53405$ & $16.77809$ \\
\hline
\end{tabular})



