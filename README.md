# X-Trans2Cap
**[CVPR2022]** X-Trans2Cap: Cross-Modal Knowledge Transfer using Transformer for 3D Dense Captioning [[Arxiv Paper]](https://arxiv.org/abs/2203.00843)

![](figures/pipeline.png)

## Citation

If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{yuan2022x,
  title={X-Trans2Cap: Cross-Modal Knowledge Transfer using Transformer for 3D Dense Captioning},
  author={Yuan, Zhihao and Yan, Xu and Liao, Yinghong and Guo, Yao and Li, Guanbin and Li, Zhen and Cui, Shuguang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Data

### ScanRefer

If you would like to access to the ScanRefer dataset, please fill out [this form](https://forms.gle/aLtzXN12DsYDMSXX6). Once your request is accepted, you will receive an email with the download link.

> Note: In addition to language annotations in ScanRefer dataset, you also need to access the original ScanNet dataset. Please refer to the [ScanNet Instructions](data/scannet/README.md) for more details.

Download the dataset by simply executing the wget command:
```shell
wget <download_link>
```

Run this commoand to organize the ScanRefer data:
```bash
python scripts/organize_data.py
```

### Processed 2D Features
You can download the processed 2D Image features from [OneDrive](https://cuhko365-my.sharepoint.com/:u:/g/personal/221019046_link_cuhk_edu_cn/EYoVKnDvr89OoWstNIK2aDEBWjBmxAovQjg6bP34xZ3j2w?e=zvGRom). The feature extraction code is borrowed from [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch).

Change the data path in `lib/config.py`.

## Training

Run this command to train the model:

```bash
python train.py --config config/xtrans_scanrefer.yaml
```