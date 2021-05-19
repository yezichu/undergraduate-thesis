from __future__ import print_function, division
import numpy as np
import SimpleITK as sitk
import os

flair_name = "_flair.nii.gz"
t1_name = "_t1.nii.gz"
t1ce_name = "_t1ce.nii.gz"
t2_name = "_t2.nii.gz"
mask_name = "_seg.nii.gz"

train_brats_path = "/home/mimt/jupyter/wy/train"
output_trainImage = "/home/mimt/jupyter/wy/trainImage"
output_trainMask = "/home/mimt/jupyter/wy/trainMask"

Validation_brats_path = "/home/mimt/jupyter/wy/validate"
output_ValidationImage = "/home/mimt/jupyter/wy/validationImage"
output_ValidationMask = "/home/mimt/jupyter/wy/validationMask"

BLOCKSIZE = (80, 80, 80)  # 每个分块的大小


def crop_ceter(img, croph, cropw):
    height, width = img[0].shape
    starth = height // 2 - (croph // 2)
    startw = width // 2 - (cropw // 2)
    return img[:, starth:starth + croph, startw:startw + cropw]


def normalize(slice, bottom=99, down=1):
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        tmp[tmp == tmp.min()] = -9
        return tmp


def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files


def part(brats_path, path_list, nameImage, nameMask):
    for subsetindex in range(len(path_list)):
        print(path_list[subsetindex])
        # 读取数据
        brats_subset_path = brats_path + "/" + str(
            path_list[subsetindex]) + "/"
        flair_image = brats_subset_path + str(
            path_list[subsetindex]) + flair_name
        t1_image = brats_subset_path + str(path_list[subsetindex]) + t1_name
        t1ce_image = brats_subset_path + str(
            path_list[subsetindex]) + t1ce_name
        t2_image = brats_subset_path + str(path_list[subsetindex]) + t2_name
        mask_image = brats_subset_path + str(
            path_list[subsetindex]) + mask_name
        flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
        t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
        t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
        t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
        mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
        flair_array = sitk.GetArrayFromImage(flair_src)
        t1_array = sitk.GetArrayFromImage(t1_src)
        t1ce_array = sitk.GetArrayFromImage(t1ce_src)
        t2_array = sitk.GetArrayFromImage(t2_src)
        mask_array = sitk.GetArrayFromImage(mask)
        # 人工加入切片
        myblackslice = np.zeros([240, 240])
        flair_array = np.insert(flair_array, 0, myblackslice, axis=0)
        flair_array = np.insert(flair_array, 0, myblackslice, axis=0)
        flair_array = np.insert(flair_array, 0, myblackslice, axis=0)
        flair_array = np.insert(flair_array,
                                flair_array.shape[0],
                                myblackslice,
                                axis=0)
        flair_array = np.insert(flair_array,
                                flair_array.shape[0],
                                myblackslice,
                                axis=0)
        t1_array = np.insert(t1_array, 0, myblackslice, axis=0)
        t1_array = np.insert(t1_array, 0, myblackslice, axis=0)
        t1_array = np.insert(t1_array, 0, myblackslice, axis=0)
        t1_array = np.insert(t1_array, t1_array.shape[0], myblackslice, axis=0)
        t1_array = np.insert(t1_array, t1_array.shape[0], myblackslice, axis=0)
        t1ce_array = np.insert(t1ce_array, 0, myblackslice, axis=0)
        t1ce_array = np.insert(t1ce_array, 0, myblackslice, axis=0)
        t1ce_array = np.insert(t1ce_array, 0, myblackslice, axis=0)
        t1ce_array = np.insert(t1ce_array,
                               t1ce_array.shape[0],
                               myblackslice,
                               axis=0)
        t1ce_array = np.insert(t1ce_array,
                               t1ce_array.shape[0],
                               myblackslice,
                               axis=0)
        t2_array = np.insert(t2_array, 0, myblackslice, axis=0)
        t2_array = np.insert(t2_array, 0, myblackslice, axis=0)
        t2_array = np.insert(t2_array, 0, myblackslice, axis=0)
        t2_array = np.insert(t2_array, t2_array.shape[0], myblackslice, axis=0)
        t2_array = np.insert(t2_array, t2_array.shape[0], myblackslice, axis=0)
        mask_array = np.insert(mask_array, 0, myblackslice, axis=0)
        mask_array = np.insert(mask_array, 0, myblackslice, axis=0)
        mask_array = np.insert(mask_array, 0, myblackslice, axis=0)
        mask_array = np.insert(mask_array,
                               mask_array.shape[0],
                               myblackslice,
                               axis=0)
        mask_array = np.insert(mask_array,
                               mask_array.shape[0],
                               myblackslice,
                               axis=0)

        # 分成200，200，200
        i = 0
        while i < 20:
            flair_array = np.insert(flair_array, 0, myblackslice, axis=0)
            flair_array = np.insert(flair_array,
                                    flair_array.shape[0],
                                    myblackslice,
                                    axis=0)
            t1_array = np.insert(t1_array, 0, myblackslice, axis=0)
            t1_array = np.insert(t1_array,
                                 t1_array.shape[0],
                                 myblackslice,
                                 axis=0)
            t1ce_array = np.insert(t1ce_array, 0, myblackslice, axis=0)
            t1ce_array = np.insert(t1ce_array,
                                   t1ce_array.shape[0],
                                   myblackslice,
                                   axis=0)
            t2_array = np.insert(t2_array, 0, myblackslice, axis=0)
            t2_array = np.insert(t2_array,
                                 t2_array.shape[0],
                                 myblackslice,
                                 axis=0)
            mask_array = np.insert(mask_array, 0, myblackslice, axis=0)
            mask_array = np.insert(mask_array,
                                   mask_array.shape[0],
                                   myblackslice,
                                   axis=0)
            i += 1
        # 对四个模态分别进行标准化
        flair_array_nor = normalize(flair_array)
        t1_array_nor = normalize(t1_array)
        t1ce_array_nor = normalize(t1ce_array)
        t2_array_nor = normalize(t2_array)
        # 裁剪
        flair_crop = crop_ceter(flair_array_nor, 200, 200)
        t1_crop = crop_ceter(t1_array_nor, 200, 200)
        t1ce_crop = crop_ceter(t1ce_array_nor, 200, 200)
        t2_crop = crop_ceter(t2_array_nor, 200, 200)
        mask_crop = crop_ceter(mask_array, 200, 200)

        # 5、分块处理
        patch_block_size = BLOCKSIZE
        numberxy = 60
        numberz = 60
        width = np.shape(flair_crop)[1]
        height = np.shape(flair_crop)[2]
        imagez = np.shape(flair_crop)[0]
        block_width = np.array(patch_block_size)[1]
        block_height = np.array(patch_block_size)[2]
        blockz = np.array(patch_block_size)[0]
        stridewidth = (width - block_width) // numberxy
        strideheight = (height - block_height) // numberxy
        stridez = (imagez - blockz) // numberz
        step_width = width - (stridewidth * numberxy + block_width)
        step_width = step_width // 2
        step_height = height - (strideheight * numberxy + block_height)
        step_height = step_height // 2
        step_z = imagez - (stridez * numberz + blockz)
        step_z = step_z // 2
        hr_samples_flair_list = []
        hr_samples_t1_list = []
        hr_samples_t1ce_list = []
        hr_samples_t2_list = []
        hr_mask_samples_list = []
        patchnum = []
        for z in range(step_z, numberz * (stridez + 1) + step_z, numberz):
            for x in range(step_width,
                           numberxy * (stridewidth + 1) + step_width,
                           numberxy):
                for y in range(step_height,
                               numberxy * (strideheight + 1) + step_height,
                               numberxy):
                    patchnum.append(z)
                    hr_samples_flair_list.append(
                        flair_crop[z:z + blockz, x:x + block_width,
                                   y:y + block_height])
                    hr_samples_t1_list.append(t1_crop[z:z + blockz,
                                                      x:x + block_width,
                                                      y:y + block_height])
                    hr_samples_t1ce_list.append(t1ce_crop[z:z + blockz,
                                                          x:x + block_width,
                                                          y:y + block_height])
                    hr_samples_t2_list.append(t2_crop[z:z + blockz,
                                                      x:x + block_width,
                                                      y:y + block_height])
                    hr_mask_samples_list.append(mask_crop[z:z + blockz,
                                                          x:x + block_width,
                                                          y:y + block_height])
        samples_flair = np.array(hr_samples_flair_list).reshape(
            (len(hr_samples_flair_list), blockz, block_width, block_height))
        samples_t1 = np.array(hr_samples_t1_list).reshape(
            (len(hr_samples_t1_list), blockz, block_width, block_height))
        samples_t1ce = np.array(hr_samples_t1ce_list).reshape(
            (len(hr_samples_t1ce_list), blockz, block_width, block_height))
        samples_t2 = np.array(hr_samples_t2_list).reshape(
            (len(hr_samples_t2_list), blockz, block_width, block_height))
        mask_samples = np.array(hr_mask_samples_list).reshape(
            (len(hr_mask_samples_list), blockz, block_width, block_height))
        samples, imagez, height, width = np.shape(samples_flair)[0], np.shape(
            samples_flair)[1], np.shape(samples_flair)[2], np.shape(
                samples_flair)[3]
        for j in range(samples):
            """
            merage 4 model image into 4 channel (imagez,width,height,channel)
            """
            fourmodelimagearray = np.zeros((imagez, height, width, 4),
                                           np.float)
            filepath1 = nameImage + "/" + path_list[subsetindex] + "_" + str(
                j).zfill(2) + ".npy"
            filepath = nameMask + "/" + path_list[subsetindex] + "_" + str(
                j).zfill(2) + ".npy"
            flairimage = samples_flair[j, :, :, :]
            flairimage = flairimage.astype(np.float)
            fourmodelimagearray[:, :, :, 0] = flairimage
            t1image = samples_t1[j, :, :, :]
            t1image = t1image.astype(np.float)
            fourmodelimagearray[:, :, :, 1] = t1image
            t1ceimage = samples_t1ce[j, :, :, :]
            t1ceimage = t1ceimage.astype(np.float)
            fourmodelimagearray[:, :, :, 2] = t1ceimage
            t2image = samples_t2[j, :, :, :]
            t2image = t2image.astype(np.float)
            fourmodelimagearray[:, :, :, 3] = t2image
            np.save(filepath1, fourmodelimagearray)

            wt_tc_etMaskArray = np.zeros((imagez, height, width, 3), np.uint8)
            mask_one_sample = mask_samples[j, :, :, :]
            WT_Label = mask_one_sample.copy()
            WT_Label[mask_one_sample == 1] = 1.
            WT_Label[mask_one_sample == 2] = 1.
            WT_Label[mask_one_sample == 4] = 1.
            TC_Label = mask_one_sample.copy()
            TC_Label[mask_one_sample == 1] = 1.
            TC_Label[mask_one_sample == 2] = 0.
            TC_Label[mask_one_sample == 4] = 1.
            ET_Label = mask_one_sample.copy()
            ET_Label[mask_one_sample == 1] = 0.
            ET_Label[mask_one_sample == 2] = 0.
            ET_Label[mask_one_sample == 4] = 1.
            wt_tc_etMaskArray[:, :, :, 0] = WT_Label
            wt_tc_etMaskArray[:, :, :, 1] = TC_Label
            wt_tc_etMaskArray[:, :, :, 2] = ET_Label
            np.save(filepath, wt_tc_etMaskArray)
    print("Done!")


brats_path = train_brats_path
trainImage = output_trainImage
trainMask = output_trainMask

if not os.path.exists(trainImage):
    os.makedirs(output_trainImage)
    print("trainImage 输出目录创建成功")
if not os.path.exists(trainMask):
    os.makedirs(trainMask)
    print("trainMask 输出目录创建成功")

brats_path1 = Validation_brats_path
ValidationImage = output_ValidationImage
ValidationMask = output_ValidationMask
if not os.path.exists(ValidationImage):
    os.makedirs(output_ValidationImage)
    print("ValidationImage 输出目录创建成功")
if not os.path.exists(ValidationMask):
    os.makedirs(ValidationMask)
    print("ValidationMask 输出目录创建成功")

path_list = file_name_path(brats_path)
path_list1 = file_name_path(brats_path1)

part(brats_path, path_list, trainImage, trainMask)
part(brats_path1, path_list1, ValidationImage, ValidationMask)
