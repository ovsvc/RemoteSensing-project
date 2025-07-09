# RemoteSensing-project

Repository for Interdisciplinary Project in Data Science

**Topic:** Landsat and Sentinel-2 based floods maps of the Central EuropeanFlood event in September 2024

**Author:** Viktoriia Ovsianik (12217985)

### 1. Project Summary
This project aims to **develop a machine learning-based approach for assessing floods using satellite imagery**, with a focus on the Central European flood event of September 2024. Leveraging multispectral optical imagery from the Landsat and Sentinel-2 satellite series—including RGB and near-infrared (NIR) bands—the project seeks to **estimate flood probability at the pixel level, rather than relying on traditional binary classification methods**. 

The core of the methodology involves training a convolutional neural network for semantic segmentation (DeepLabV3+) to detect water-covered areas, extended with Monte Carlo dropout to generate uncertainty-aware predictions. The workflow includes data collection, preprocessing (e.g., cloud masking, tiles splitting), and feature engineering (e.g., NDWI), followed by model training and evaluation using standard metrics such as IoU, F1 score, Precision, Recall, and Dice Coefficient.

Ultimately, the project aims to assess the utility of optical satellite data in flood mapping and demonstrate the model’s ability to distinguish floodwaters from other land and water surfaces with quantifiable confidence.

------------

### 2. Dataset
This project includes a custom-built dataset of satellite images from Sentinel-2 and Landsat 8/9 for water and flood segmentation. 


#### 2.1 Dataset Collection
▶️ Relevant code for dataset collection part - `00_creating_Landsat_dataset.ipynb` and `00_creating_Sentinel_dataset.ipynb`.
Geojsons relevant for every sampling strategy and flood event can be found in `geojsons`.

Satellite images for the dataset were extracted from [Sentinel Hub](https://www.sentinel-hub.com/) using stratified random sampling strategy. Data was extracted using 3 strategies: 
* General sampling: images randomly selected across Central & Eastern Europe (2022–2024)
* Waterbody sampling: images extracted from river-rich areas
* Flood event sampling: post-event imagery from major floods (Storm Boris (Sep 2024), Bavaria Flood (May–Jun 2024), Germany Flood (Jul–Aug 2021))



#### 2.2 Mask Annotation

▶️ Relevant code for masks annotation part - `01_correcting masks.ipynb` and `02_splitting_Sentinel_dataset.ipynb`.

The next step after dataset collection was to create water masks:
* Sentinel-2:

Water masks were created from NDWI thresholding + SCL water class. Index-based masks were checked manually, corrections, where necessary, were applied using `napari` Python libtrary.

* Landsat:

Water/ice classified using NDWI + NDSI thresholds. Index-based masks were extended by adding permanent water pixels from JRC GSW dataset.


#### 2.3  Tiling

▶️ Relevant code for tiling part - `02_splitting_Sentinel_dataset.ipynb`.

Images & corresponding masks were split into 256×256 pixel tiles. Cloud-dominated tiles were excluded. Additionally, Landsat images were resampled to 10m resolution to match Sentinel-2.

#### 2.4  Dataset Balancing & JupiterHub Uploading

▶️ Relevant code for balancing & JupiterHub Uploading part - `02_splitting_Sentinel_dataset.ipynb`.

Strong class imbalance was detected for Sentinel dataset (water < 2% of all pixels), the issue was addressed by oversampling water-rich tiles.
Final dataset includes ~16,000 tiles with ~10% water pixels, the dataset was split into train (70%), val (15%), test (15%). Flood-specific images (e.g., Storm Boris) held out for evaluation only.

All experiments with models were conducted in JupiterHub (T4 GPU), to upload dataset to JupiterHub additional pipeline was created. 

------------
### 2. Modelling



As the project was conducted in collaboration with other students there were 3 different approaches to modelling:
* Training on Sentinel-2 dataset → testing on Sentinel-2 & Landsat datasets
* Training on Landsat dataset → testing on Sentinel-2 & Landsat datasets
* Training on Sentinel-2 & Landsat datasets → testing on Sentinel-2 & Landsat datasets

#### 2.1 Training on Sentinel-2 dataset → testing on Sentinel-2 & Landsat datasets

▶️ Relevant code for modelling part - `03_Modelling.ipynb`.

**2.1.1 Final model** 

DeepLabV3 + ResNet34 for Water Segmentation with Uncertainty Estimation (Monte Carlo Dropout).
Model provides classification into 4 classes — background, water, cloud, and snow/ice.
The schema is based on the original paper by Chen et al.

![Slide1](https://github.com/user-attachments/assets/1e19c1d3-c4a2-472a-be50-fd4c16be1983)

**2.1.2 Training Setup**

* Loss Function: 0.5 x Weighted Cross-Entropy Loss ([0.1, 0.9, 0.3, 0.3]) +  0.5 × Dice

* Optimizer: AdamW (LR = 0.001, weight decay = 1e-4)

* Scheduler: ReduceLROnPlateau with patience of 3

* Training Config: Batch size - 128; Epochs - 14; Early stopping - patience = 3

* Training subset: 8,000 samples

**2.1.3 Evaluation**

* The following metrics were used for evaluation: Precision, Recall, IoU, Dice, Accuracy, mIoU, Macro Dice.

* Special focus was placed on class 1 (water), as flood detection was the primary goal.

* Experiment tracking via [Weights & Biases ](https://wandb.ai/home)

**2.3.4 Experimentation Highlights**

Before defining the final model, multiple architectures and their parameters were tested - U-Net (with ResNet-50, ResNet-34, MobileNetV2), DeepLabV3 with different backbones, additionally, multiple loss strategies and augmentation pipelines were explored. 

