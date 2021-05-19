This is the code for my graduation thesis and the solution for my BRATS 2020 Challenge.

# Image processing
![image](/image/data_sample.png)
Here are the four modes of the image and its mask

# Neural network architecture
![4](/image/net图片.jpg)

# Training

First,the image is cropped into patches.Run`srrunc/partition.ipy`,change the storage address of training set, verification set and test set successively.

```python
train_brats_path = "your-Path_to/MICCAI_BraTS_2020_Data_Training"
output_trainImage = "your-Path_to/trainImage"
output_trainMask = "your-Path_to/trainMask"
```
And run`src/train.py`，change data_path.Began to run

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

**训练集评价指标**
<table class=MsoTable15Plain2 border=1 cellspacing=0 cellpadding=0
 style='border-collapse:collapse;border:none;mso-border-top-alt:solid #7F7F7F .5pt;
 mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
 solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
 128;mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:-1;mso-yfti-firstrow:yes;mso-yfti-lastfirstrow:yes'>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:5'><b><span
  lang=EN-US><o:p>&nbsp;</o:p></span></b></p>
  </td>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:1'><b><span
  lang=EN-US>ET<o:p></o:p></span></b></p>
  </td>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:1'><b><span
  lang=EN-US>WT<o:p></o:p></span></b></p>
  </td>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:1'><b><span
  lang=EN-US>TC<o:p></o:p></span></b></p>
  </td>
  <td width=111 style='width:83.0pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:1'><b><span
  lang=EN-US>Mean<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:0'>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;mso-border-top-alt:solid #7F7F7F .5pt;
  mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
  solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
  128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:68'><b><span
  lang=EN-US>Dice<o:p></o:p></span></b></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;mso-border-top-alt:solid #7F7F7F .5pt;
  mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
  solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
  128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.79655</span></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;mso-border-top-alt:solid #7F7F7F .5pt;
  mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
  solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
  128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.93</span></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;mso-border-top-alt:solid #7F7F7F .5pt;
  mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
  solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
  128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.90825</span></p>
  </td>
  <td width=111 style='width:83.0pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;mso-border-top-alt:solid #7F7F7F .5pt;
  mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
  solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
  128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.878267</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1'>
  <td width=111 style='width:82.95pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:4'><b><span
  lang=EN-US>Sensitivity<o:p></o:p></span></b></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>0.78583</span></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>0.91073</span></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>0.91593</span></p>
  </td>
  <td width=111 style='width:83.0pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>0.87083</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2'>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:68'><b><span
  lang=EN-US>Specificity<o:p></o:p></span></b></p>
  </td>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.99978</span></p>
  </td>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.99959</span></p>
  </td>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.9996</span></p>
  </td>
  <td width=111 style='width:83.0pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.999657</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3;mso-yfti-lastrow:yes'>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-bottom-alt:solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;
  mso-border-bottom-themetint:128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:4'><b><span
  lang=EN-US>Hausdorff95<o:p></o:p></span></b></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-bottom-alt:solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;
  mso-border-bottom-themetint:128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>22.91558</span></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-bottom-alt:solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;
  mso-border-bottom-themetint:128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>3.84997</span></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-bottom-alt:solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;
  mso-border-bottom-themetint:128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>3.74955</span></p>
  </td>
  <td width=111 style='width:83.0pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-bottom-alt:solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;
  mso-border-bottom-themetint:128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>10.1717</span></p>
  </td>
 </tr>
</table>

**验证集评价指标**
<table class=MsoTable15Plain2 border=1 cellspacing=0 cellpadding=0
 style='border-collapse:collapse;border:none;mso-border-top-alt:solid #7F7F7F .5pt;
 mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
 solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
 128;mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:-1;mso-yfti-firstrow:yes;mso-yfti-lastfirstrow:yes'>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:5'><b><span
  lang=EN-US><o:p>&nbsp;</o:p></span></b></p>
  </td>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:1'><b><span
  lang=EN-US>ET<o:p></o:p></span></b></p>
  </td>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:1'><b><span
  lang=EN-US>WT<o:p></o:p></span></b></p>
  </td>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:1'><b><span
  lang=EN-US>TC<o:p></o:p></span></b></p>
  </td>
  <td width=111 style='width:83.0pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:1'><b><span
  lang=EN-US>Mean<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:0'>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;mso-border-top-alt:solid #7F7F7F .5pt;
  mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
  solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
  128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:68'><b><span
  lang=EN-US>Dice<o:p></o:p></span></b></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;mso-border-top-alt:solid #7F7F7F .5pt;
  mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
  solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
  128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.71534</span></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;mso-border-top-alt:solid #7F7F7F .5pt;
  mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
  solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
  128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.89413</span></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;mso-border-top-alt:solid #7F7F7F .5pt;
  mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
  solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
  128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.80496</span></p>
  </td>
  <td width=111 style='width:83.0pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;mso-border-top-alt:solid #7F7F7F .5pt;
  mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
  solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
  128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.80481</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1'>
  <td width=111 style='width:82.95pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:4'><b><span
  lang=EN-US>Sensitivity<o:p></o:p></span></b></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>0.70984</span></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>0.88827</span></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>0.81775</span></p>
  </td>
  <td width=111 style='width:83.0pt;border:none;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>0.805287</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2'>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:68'><b><span
  lang=EN-US>Specificity<o:p></o:p></span></b></p>
  </td>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.99971</span></p>
  </td>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.99917</span></p>
  </td>
  <td width=111 style='width:82.95pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.99938</span></p>
  </td>
  <td width=111 style='width:83.0pt;border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:
  text1;mso-border-top-themetint:128;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:64'><span
  lang=EN-US>0.99942</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3;mso-yfti-lastrow:yes'>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-bottom-alt:solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;
  mso-border-bottom-themetint:128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:4'><b><span
  lang=EN-US>Hausdorff95<o:p></o:p></span></b></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-bottom-alt:solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;
  mso-border-bottom-themetint:128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>36.16794</span></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-bottom-alt:solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;
  mso-border-bottom-themetint:128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>4.63229</span></p>
  </td>
  <td width=111 style='width:82.95pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-bottom-alt:solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;
  mso-border-bottom-themetint:128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>9.53405</span></p>
  </td>
  <td width=111 style='width:83.0pt;border:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-bottom-alt:solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;
  mso-border-bottom-themetint:128;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US>16.77809</span></p>
  </td>
 </tr>
</table>





