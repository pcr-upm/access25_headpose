# Reducing head pose estimation data set bias with synthetic data (IEEE Access 2025)
If you use this code for your own research, you must reference our paper:
```
Reducing head pose estimation data set bias with synthetic data
Roberto Valle, José M. Buenaposada, Luis Baumela.
IEEE Access 2025.
https://doi.org/10.1109/ACCESS.2025.3561506
```

#### Requisites
- images-framework
- torch
- pytorch-lightning
- torchvision
- torch-summary
- tensorboard
- tqdm
- scikit-learn

#### Usage
```
usage: access25_headpose_test.py [-h] [--input-data INPUT_DATA] [--show-viewer] [--save-image]
```

* Use the --input-data option to set an image, directory, camera or video file as input.

* Use the --show-viewer option to show results visually.

* Use the --save-image option to save the processed images.
```
usage: Alignment --database DATABASE
```

* Use the --database option to select the database model.
```
usage: Access25Headpose [--gpu GPU] --backbone {resnet,efficientnet} [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--patience PATIENCE]
```

* Use the --gpu option to set the GPU identifier (negative value indicates CPU mode).

* Use the --backbone option to set the deep architecture.

* Use the --batch-size option to set the number of images in each mini-batch.

* Use the --epochs option to set the number of sweeps over the dataset to train.

* Use the --patience option to set number of epochs with no improvement after which training will be stopped.
```
> python test/access25_headpose_test.py --input-data test/example.tif --database aflw --gpu 0 --backbone resnet --save-image
```