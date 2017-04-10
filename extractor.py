from model.resnet50 import ResNet50

def get_extractor():
    model = ResNet50(include_top=False, weights='imagenet')
    return model
