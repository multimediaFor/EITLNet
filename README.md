## EFFECTIVE IMAGE TAMPERING LOCALIZATION VIA ENHANCED TRANSFORMER AND CO-ATTENTION FUSION 
### Network Architecture
![EITLNet](./EITLNet.png)



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

### Trained Models
Please download the models and place them in the [weights](weights) directory:
+ [weight](https://www.123pan.com/s/2pf9-0tPHv.html) (password：EITL)

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
