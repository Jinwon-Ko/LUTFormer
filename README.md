# LUTFormer: Lookup Table Transformer for Image Enhancement.
### Jinwon Ko, Keunsoo Ko, Hanul Kim and Chang-Su Kim.

Official code for **"LUTFormer: Lookup Table Transformer for Image Enhancement"**.

### Dataset


### Installation
Create conda environment:
```
$ conda create -n LUTFormer python=3.9 anaconda
$ conda activate LUTFormer
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
$ pip install opencv-python-headless==4.10.0.82
```

### Train
To train LUTFormer,
1. Edit `root/LUTFormer_code/config.py`. Please modify `run_mode` to `'train'`. Also, set the `task_name`, `dataset_name`, and `expert`. If you want to visualize the results, set `viz` to `True`.
2. Run with
```
$ cd root/LUTFormer_code/
$ python main.py
```

### Test
To test your trained LUTFormer model,
1. Edit `root/LUTFormer_code/config.py`. Please modify `run_mode` to `'test'`. Also, set the `task_name`, `dataset_name`, and `expert`.
2. Run with
```
$ cd root/LUTFormer_code/
$ python main.py
```

If you want to get the performance of the paper, download our [pre-trained model](https://drive.google.com/file/d/1X70k12VlxTus5ppQlq-ZPl53zxSv-uom/view?usp=sharing). to ```root/pretrained/```.
1. Edit `root/LUTFormer_code/config.py`. Please modify `run_mode` to `'test_paper'`. Also, set the `task_name`, `dataset_name`, and `expert`.
2. Run with
```
cd root/LUTFormer_code/
python main.py
```



### Results
1. Photo retouching on FiveK dataset
<img src="https://github.com/Jinwon-Ko/LUTFormer/blob/main/results/Results_for_Retouching_FiveK.jpg" alt="Retouching FiveK" width="100%" height="70%" border="10"/>


2. Photo retouching on PPR10K dataset
<img src="https://github.com/Jinwon-Ko/LUTFormer/blob/main/results/Results_for_Retouching_PPR10K.jpg" alt="Retouching PPR10K" width="80%" height="70%" border="10"/>


3. Tone mapping on FiveK dataset
<img src="https://github.com/Jinwon-Ko/LUTFormer/blob/main/results/Results_for_ToneMap_FiveK.jpg" alt="ToneMap FiveK" width="100%" height="70%" border="10"/>


4. Underwater image enhancement on UIEB dataset
<img src="https://github.com/Jinwon-Ko/LUTFormer/blob/main/results/Results_for_Underwater_UIEB.jpg" alt="Underwater UIEB" width="100%" height="70%" border="10"/>

