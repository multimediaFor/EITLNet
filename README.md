## EFFECTIVE IMAGE TAMPERING LOCALIZATION VIA ENHANCED TRANSFORMER
AND CO-ATTENTION FUSION 

### Network Architecture
![image-20240126201127386](C:\Users\guokun\AppData\Roaming\Typora\typora-user-images\image-20240126201127386.png)



### Environment

- Python 3.8
- cuda11.1+cudnn8.0.4

### Requirements

- pip install requirements.txt

### Training datasets

The training dataset catalog is as follows.

```
├─train_dataset
    ├─ImageSets
    │  └─Segmentation
    │          train.txt
    │          val.txt
    ├─JPEGImages
    │      00001.jpg
    │      00002.jpg
    │      00003.jpg│      
    │      ...
    └─SegmentationClass
            00001_gt.png
            00002_gt.png
            00003_gt.png
```



### Training
```python
python train.py
```

### Testing

```
python test.py
```

## Bibtex
 ```
@article{guo2023effective,
  title={Effective Image Tampering Localization via Enhanced Transformer and Co-attention Fusion},
  author={Guo, Kun and Zhu, Haochen and Cao, Gang},
  journal={arXiv preprint arXiv:2309.09306},
  year={2023}
}
 ```
### Contact

If you have any questions, please contact me(guokun21@qq.com).