import torchvision.models as models
from detectron2.modeling import Backbone
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY



# define vgg16 backbone
class VGG16Backbone(Backbone):
    def __init__(self, vgg_model):
        super().__init__()
        self.features = vgg_model.features  # Use the feature extraction part of VGG16
        
    def forward(self, x):
        # Implement the forward pass
        x = self.features(x)
        return x
    
    def output_shape(self):
        # Implement the method to inform Detectron2 the output shape of your backbone
        return {"vgg16": ShapeSpec(channels=512, stride=32)}  


    # register vgg16 backbone
    @BACKBONE_REGISTRY.register()
    def build_vgg_backbone(cfg, input_shape):
        # Initialize your backbone with the pre-trained model
        model = models.vgg16(pretrained=True)
        backbone = VGG16Backbone(model)
        return backbone
    