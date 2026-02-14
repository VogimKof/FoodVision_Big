### Description

### Project overview
### Datasets 
### Model architecture

### App preview
Available (or not) at https://huggingface.co/spaces/VogimKof/FoodVision_Big

### Technologies 

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EA4335?logo=pytorch&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Interface](https://img.shields.io/badge/UI-Gradio-ff69b4)

### Repo structure
```
FoodVision_Big/
├── configs/
│   ├── food_classes.txt  -> names of food classes are stored here
│   └── nonfood_classes.txt -> names of non-food classes are stored here
├── data/
│   └── examples/ -> example images to use in demo app
├── demos/
│   └── app.py -> demo app interface made with gradio 
├── func/
│   ├── data_setup.py -> scripts to create, split and load images into datasets and pytorch dataloaders
│   ├── engine.py -> scripts to perform model training 
│   ├── get_data.py -> scripts to download images 
│   ├── model_setup.py -> scripts to create EfficientNetB2 model
│   ├── utils.py -> utility scripts to save models, create SummaryWriter and perform sample predictions
│   └── visuals.py -> scripts to visualize model performance and image transformations
├── models/ -> stores model weights 
├── notebooks/
│   ├── Food type classifier prep.ipynb
│   └── Food_NonFood classifier prep.ipynb
├── pyproject.toml
└── README.md
```

### things to improve itd.


