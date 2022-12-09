from fastai.vision.all import *
from fastai.vision.widgets import *



DATASET_PATH = Path ('datasets/train')
models_list = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet50_32x4d": models.resnext50_32x4d,
    "efficient_net_b0": models.efficientnet_b0,
}

def fastai_process(model, file_name):
    drowsiness_datablock = DataBlock(
    get_items = get_image_files,
    get_y = parent_label,
    blocks = (ImageBlock, CategoryBlock),
    item_tfms = RandomResizedCrop(224, min_scale = 0.3),
    splitter = RandomSplitter(valid_pct = 0.2, seed = 100),
    batch_tfms = aug_transforms(mult = 3, min_scale=0.8)
    )  
    dls = drowsiness_datablock.dataloaders(DATASET_PATH, num_workers=0)
    learn = vision_learner(dls, model, metrics = accuracy, pretrained=True)

    lr = learn.lr_find()
    
    learn.fit_one_cycle(20, lr)

    learn.recorder.plot_loss()    
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    
    interp.plot_top_losses(15, nrows = 15)

    learn.export("learners/" + file_name + ".pkl")

def main():
    for model_name, model in models_list.items():
        print(f"Start the process of model {model_name}")
        fastai_process(model, model_name)
        print(f"End of the model {model_name} training")

if __name__ == "__main__":
    main()