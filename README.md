
<h1 align="center">[CLS] and [EOS] Tokens are not Enough for CLIP-based Class-Incremental Learning</h1>

<div align="center">
    <div>
        Hao Sun<sup>1</sup>&emsp;
        Tao Hu<sup>1</sup>&emsp;
        Shixiong Kai<sup>2</sup>&emsp;
        Siyuan Xu<sup>2</sup>&emsp;
        Wenqian Zhao<sup>2</sup>&emsp;
        Mingxuan Yuan<sup>2</sup>&emsp;
        Da-Wei Zhou<sup>1</sup>
    </div>
    <div>
        <sup>1</sup>School of Artificial Intelligence and State Key Laboratory for Novel Software Technology, Nanjing University
    </div>
    <div>
        <sup>2</sup>Huawei Noah's Ark Lab
    </div>
</div>

The official PyTorch implementation of **[CLS] and [EOS] Tokens are not Enough for CLIP-based Class-Incremental Learning**.

If you find this repository useful in your research, please consider citing our paper:


```bibtex
@article{Sun26pace,
  author       = {Hao Sun and Tao Hu and Shixiong Kai and Siyuan Xu and Wenqian Zhao and Mingxuan Yuan and Da-Wei Zhou},
  title        = {[CLS] and [EOS] Tokens are not Enough for CLIP-based Class-Incremental Learning},
  journal      = {IEEE Transactions on Multimedia},
  year         = {2026},
}
```



# 📢 **Updates**

[10/2026] Accepted to [TMM 2026](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6046).

[11/2026] Code has been released.


# 📝 Introduction
Class-Incremental Learning (CIL) aims to empower models to continuously acquire new knowledge while mitigating catastrophic forgetting. With the advent of vision-language models like CLIP, leveraging their cross-modal alignment capabilities to address CIL challenges has emerged as a dominant paradigm. Existing methods leverage CLIP's global representations (*i.e.*, the visual [CLS] token and the textual [EOS] token) for cross-modal alignment. While achieving effective classification, they overlook the rich patch-level visual features and token-level textual semantics inherent in CLIP. For example, the [CLS] token aggregates local information to capture the general concept of "cat", while the patch-level details retain specific localized features essential for recognition, such as the tail, fur, and face. We propose **PACE (PAtch-token Cross-modal Enhancement)** to uncover the long-neglected local representations within CLIP. First, based on the global representations like [CLS] token and [EOS] token, we identify the top-*k* most crucial visual patches and textual tokens. Subsequently, to better utilize the extracted key local features, we design a cross-modal fusion module. This module leverages local textual semantics to refine visual representations, while augmenting textual semantics with local visual features. As the fusion module is shared across all tasks, directly fine-tuning it on new tasks would interfere with the knowledge learned from previous tasks, leading to catastrophic forgetting. To alleviate this, we employ a projection-based update strategy to effectively preserve the critical parameter space of previous tasks during the adaptation to new tasks. Extensive experiments demonstrate that PACE significantly outperforms state-of-the-art methods across multiple CIL benchmarks.


## 🔧 Requirements

**Environment**

1 [torch 1.13.0](https://github.com/pytorch/pytorch)

2 [torchvision 0.15.0](https://github.com/pytorch/vision)

3 [open-clip 2.30.0](https://github.com/mlfoundations/open_clip/releases/tag/v2.30.0)

**Dataset**

We provide the processed datasets as follows:

- **CIFAR100**: will be automatically downloaded by the code.
- **CUB200**: Google Drive: [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or OneDrive [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc)
- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)
- **ObjectNet**: Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EZFv9uaaO1hBj7Y40KoCvYkBnuUZHnHnjMda6obiDpiIWw?e=4n8Kpy) You can also refer to the [filelist](https://drive.google.com/file/d/147Mta-HcENF6IhZ8dvPnZ93Romcie7T6/view?usp=sharing) and processing [code](https://github.com/zhoudw-zdw/RevisitingCIL/issues/2#issuecomment-2280462493) if the file is too large to download.
- **Cars**: Google Drive: [link](https://drive.google.com/file/d/1D8ReAuOPenWi6SMNUrOZhbm6ViyhDHbL/view?usp=sharing  ) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EbT1XAstg51Mpy82uHM0D2EBJLrtzmr_V64jeBRjqyyTnQ?e=h6g1rM)
- **UCF**: Google Drive: [link](https://drive.google.com/file/d/1Ng4w310_VDqpKbc7eYaumXTOiDxI02Wc/view?usp=sharing) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EU2qHQXjASdLh1jIl6ihZmcB6G2KvqmSw-sTlZKDE6xPbg?e=7ezvTr)
- **Aircraft**: Google Drive: [link](https://drive.google.com/file/d/1xI5r1fU0d6Nff51HuOo5w-e4sGEP46Z2/view?usp=drive_link) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/ETVliZnmPY9AvZZgcFFJ6jMB2c7TRvcq7-gso2Aqvdl_VQ?e=pWXqdP)
- **Food**: Google Drive: [link](https://drive.google.com/file/d/1rupzXpwrbxki4l-RVmsRawhz1Cm0lDY5/view?usp=drive_link) or OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/Eb4xfptD4L5Egus-SiYxrIcBDH1VewLGp4kzyACGF_Na_w?e=duA3Ia)
- **SUN**: OneDrive: [link](https://njuedu-my.sharepoint.cn/:u:/g/personal/ky2409911_365_nju_edu_cn/EcQq1-1pFulKstYtdknB4O8BGo0hnlDRarAwB4wFEgkx0Q?e=YZ0xYV)

These subsets are sampled from the original datasets. Please note that I do not have the right to distribute these datasets. If the distribution violates the license, I shall provide the filenames instead.

Set `DATA_ROOT_PATH` in [utils/data.py](utils/data.py) to your dataset directory; its default is `data/` relative to the repository root. The prepared layout is:

```text
data/
├── cifar100/
│   └── cifar-100-python/
│       ├── train
│       ├── test
│       └── meta
├── aircraft/{train,test}/
├── cars/{train,test}/
├── cub/{train,test}/
├── food101/{train,test}/
├── imagenet-r/{train,test}/
├── objectnet/{train,test}/
├── sun/{train,test}/
└── ucf/{train,test}/
```

## 💡 Running scripts

To prepare your JSON files, refer to the settings in the `exps` folder and run the following command. All main experiments from the paper are already provided in the `exps` folder, you can simply execute them to reproduce the results found in the `logs` folder.

```
python main.py --config ./exps/[configname].json
```

## 🎈 Acknowledgements

This implementation builds on [CIL_Survey](https://github.com/zhoudw-zdw/CIL_Survey), [PyCIL](https://github.com/G-U-N/PyCIL), [C3Box](https://github.com/LAMDA-CL/C3Box), and [OpenCLIP](https://github.com/mlfoundations/open_clip).


## 💭 Correspondence

If you have any questions, please  contact me via [email](mailto:sunhao@lamda.nju.edu.cn).


