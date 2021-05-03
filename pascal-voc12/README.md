# Image Segmentation trainings on Pascal VOC12

## Usage

From the host where trainml cli is installed, run
```bash
export CLEARML_API_ACCESS_KEY=...
# e.g. export CLEARML_API_ACCESS_KEY='abcdef'
export CLEARML_API_SECRET_KEY=...
# e.g. export CLEARML_API_SECRET_KEY='1234abcdef'
python submit_job.py
```

---

Code is copied from https://github.com/pytorch/ignite/tree/master/examples/references/segmentation/pascal_voc2012


## TrainML env / data setup

Results on `DEEPLEARNING_PY38` / `PASCAL VOC`
```
(base) root@198:/opt/trainml/models# pip list | grep torch
pytorch-ignite                     0.4.4
torch                              1.8.1+56b43f4
torchvision                        0.9.1
```

```
(base) root@198:/opt/trainml/models# ls -all /opt/trainml/input/
total 35
drwxr-xr-x  3  998 2020 11 Apr  6 03:31 .
drwxr-xr-x  4  998 2020  4 May  3 22:53 ..
lrwxrwxrwx  1 root root 17 Apr  6 03:31 VOC2007 -> VOCdevkit/VOC2007
lrwxrwxrwx  1 root root 22 Apr  6 03:31 VOC2007-test -> VOCdevkit/VOC2007-test
lrwxrwxrwx  1 root root 17 Apr  6 03:31 VOC2008 -> VOCdevkit/VOC2008
lrwxrwxrwx  1 root root 17 Apr  6 03:31 VOC2009 -> VOCdevkit/VOC2009
lrwxrwxrwx  1 root root 17 Apr  6 03:31 VOC2010 -> VOCdevkit/VOC2010
lrwxrwxrwx  1 root root 17 Apr  6 03:31 VOC2011 -> VOCdevkit/VOC2011
lrwxrwxrwx  1 root root 17 Apr  6 03:31 VOC2012 -> VOCdevkit/VOC2012
lrwxrwxrwx  1 root root 16 Apr  6 03:31 VOCaug -> VOCdevkit/VOCaug
drwxr-xr-x 10 root root 10 Apr  6 03:49 VOCdevkit

(base) root@198:/opt/trainml/models# ls -all /opt/trainml/input/VOCdevkit/VOCaug/dataset
total 3313
drwxr-xr-x 5 root root      9 Apr  6 03:50 .
drwxr-xr-x 4 root root      6 Apr  6 03:49 ..
drwxr-xr-x 2 root root  11357 Apr  6 03:50 cls
drwxr-xr-x 2 root root  11357 Apr  6 03:52 img
drwxr-xr-x 2 root root  11357 Apr  6 03:52 inst
-rw-r--r-- 1 root root 101976 Apr  6 03:49 train.txt
-rw-r--r-- 1 root root  67476 Apr  6 03:49 train_noval.txt
-rw-r--r-- 1 root root 136260 Apr  6 03:49 trainval.txt
-rw-r--r-- 1 root root  34284 Apr  6 03:49 val.txt
```