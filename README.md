## EFFECTIVE IMAGE TAMPERING LOCALIZATION VIA ENHANCED TRANSFORMER AND CO-ATTENTION FUSION 
### Network Architecture
![EITLNet](./EITLNet.png)

### Update

- 21.04.26, We updated the model [weights](https://www.123pan.com/s/PcP3Td-KGQod.html) (password：EITL) , the file```nets/EITLnet.py```, and the main results of the paper ( the average performance is more higher than  the paper [EITLNet](https://ieeexplore.ieee.org/abstract/document/10446332) ). The corrected experimental results are marked in <font color=Red>red</font> in the table below.

<img src="./corrected.png" alt="corrected" style="zoom:100%;" />

### Environment

- Python 3.8
- cuda11.1+cudnn8.0.4

### Requirements

- pip install requirements.txt

### Training datasets

The training dataset catalog is as follows. The mask image in the folder has only two values of 0 and 1.(```to01.py```)

```
├─train_dataset
    ├─ImageSets
    │  └─Segmentation
    │          train.txt
    │          val.txt
    ├─JPEGImages
    │      00001.jpg
    │      00002.jpg
    │      00003.jpg     
    │      ...
    └─SegmentationClass
            00001_gt.png
            00002_gt.png
            00003_gt.png
```

### Trained Models
Please download the models and place them in the [./weights](weights) directory:
+ [weights](https://www.123pan.com/s/PcP3Td-KGQod.html) (password：EITL)

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
@inproceedings{guo2023effective,
  title={Effective Image Tampering Localization via Enhanced Transformer and Co-attention Fusion},
  author={Guo, Kun and Zhu, Haochen and Cao, Gang},
  booktitle={ICASSP},
  year={2024}
}
 ```
### Contact

If you have any questions, please contact me(guokun21@qq.com).
