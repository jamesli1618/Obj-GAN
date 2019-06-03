# Obj-GAN
## Obj-GAN - Official PyTorch Implementation

Pytorch implementation for reproducing Obj-GAN results in the paper [Object-driven Text-to-Image Synthesis via Adversarial Training](https://arxiv.org/pdf/1902.10740.pdf) by Wenbo Li*, Pengchuan Zhang*, Lei Zhang, Qiuyuan Huang, Xiaodong He, Siwei Lyu, Jianfeng Gao. (This work was performed when Wenbo was an intern with Microsoft Research).

<img src="step_vis.png"/>

**Picture:** *If you are asked to draw a picture of several people in their ski gear are in the snow, chances are you will start with an outline of four persons with positioned reasonably in the center of the canvas, then add a sketch of the skis under their feet. Although it is not mentioned in the description, you may decide to add one backpack to each of them to match our common sense. Finally, you carefully finish the details, like painting their clothes blue, their scarves pink and the background white, to make the persons more realistic and the background matches the description better. To make the scene more vivid, you might sketch some brown stones in the snow which indicates that they are in mountains.*

*Now, there’s a bot that can do that, too.*

<img src="framework.png"/>

### Dependencies
python 3.6

Pytorch 0.4.1

In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:
- `python-dateutil`
- `easydict`
- `pandas`
- `torchfile`
- `nltk`
- `scikit-image`
- `spacy`
- `PyYAML`
- `cffi`
- `torchtext`
- `dill`
- `Cython`

**Data**


**Training**

**Pretrained Model**

Download and save them to `data/coco/pretrained/`
- [DAMSM for coco](https://drive.google.com/open?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ)
- [Inception v3](https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth)
- [VGG19 BN](https://download.pytorch.org/models/vgg19_bn-c79401a0.pth)
- [Box generator](https://drive.google.com/file/d/1OTZDywt1UGzUykAXBXmvVA6aAlQzbMjv/view?usp=sharing)
- [Shape generator](https://drive.google.com/file/d/1vyfXxh4eC1ccs9XNhC8OIylErhwLdvmN/view?usp=sharing)

**Sampling**

**More Results**
<img src="example.png"/>


### Citing Obj-GAN
If you find Obj-GAN useful in your research, please consider citing:

```
@article{objgan19,
  author    = {Wenbo Li, Pengchuan Zhang, Lei Zhang, Qiuyuan Huang, Xiaodong He, Siwei Lyu, Jianfeng Gao},
  title     = {Object-driven Text-to-Image Synthesis via Adversarial Training},
  Year = {2019},
  booktitle = {{CVPR}}
}
