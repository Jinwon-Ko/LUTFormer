# LUTFormer: Lookup Table Transformer for Image Enhancement.
### Jinwon Ko, Keunsoo Ko, Hanul Kim and Chang-Su Kim.

Official code for **"LUTFormer: Lookup Table Transformer for Image Enhancement"**.

### Dataset


### Installation
1. Create conda environment:
```
$ conda create -n LUTFormer python=3.9 anaconda
$ conda activate LUTFormer
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
$ pip install opencv-python==4.10.0
```

2. If you want to get the performance of the paper, download our [pre-trained model](https://drive.google.com/file/d/1X70k12VlxTus5ppQlq-ZPl53zxSv-uom/view?usp=sharing). to ```root/pretrained/```.



### Evaluation
Run with
```
cd root/code/
python main.py
```

### Train
For training LUTFormer,
1. Edit `root/code/config.py`. Please modify `run_mode` to `'train'`. Also, set the `model_name`, `dataset_name`, and `expert`.
2. Run with
```
$ cd root/code/
$ python main.py
```


### Results
1. Photo retouching on FiveK dataset
<img src="https://github.com/Jinwon-Ko/LUTFormer/blob/main/results/Results_for_Retouching_FiveK.jpg" alt="Retouching FiveK" width="100%" height="70%" border="10"/>


2. Photo retouching on PPR10K dataset
<img src="https://github.com/Jinwon-Ko/LUTFormer/blob/main/results/Results_for_Retouching_PPR10K.jpg" alt="Retouching PPR10K" width="100%" height="70%" border="10"/>


3. Tone mapping on FiveK dataset
<img src="https://github.com/Jinwon-Ko/LUTFormer/blob/main/results/Results_for_ToneMap_FiveK.jpg" alt="ToneMap FiveK" width="80%" height="70%" border="10"/>


4. Underwater image enhancement on UIEB dataset
<img src="https://github.com/Jinwon-Ko/LUTFormer/blob/main/results/Results_for_Underwater_UIEB.jpg" alt="Underwater UIEB" width="100%" height="70%" border="10"/>

