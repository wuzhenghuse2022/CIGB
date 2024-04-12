# Requirements
* Requirements
* Python 3.7 (Anaconda is recommended)
* skimage
* imageio
* Pytorch (Pytorch version >=1.2.0 is recommended)
* tqdm
* pandas
* cv2 (pip install opencv-python)

# Train
run one of following commands for evaluation:
```
python main.py --save_name CIFAR10 --dataroot ./Datasets/CIFAR10/ --dataset cifar10 --imb_factor 0.1 --lr 0.01 --weight_decay 5e-3 --batch_size 64 --idm_rate 11 --greb_rate 0.07 --gpu 0 --idx 1
```

# Test
run one of following commands for evaluation:
```
python test.py --save_name CIF10_Res32_imb10_CIGB --filename CIF10_Res32_imb10_CIGB --dataroot ./Datasets/CIFAR10/ --dataset cifar10 --imb_factor 0.1 --gpu 0 --idx 1
```
