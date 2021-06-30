import PIL
import torch
import numpy as np
from PIL import Image

class Transformer(object):
    """"Transform"""
    def __init__(self,):
        pass
    def __call__(self, img, bbox=None):
        pass

class Compose(Transformer):
    """Compose transforms"""
    def __init__(self, transforms=[]):
        super().__init__()
        self.transforms=transforms
        
    def __call__(self, img, bbox=None):
        if bbox is None:
            for transform in self.transforms:
                img = transform(img, bbox)
            return img
        for transform in self.transforms:
            img, bbox = transform(img, bbox)
        return img, bbox
    
class Resize(Transformer):
    """Resize image and bbox"""
    def __init__(self, size=(320, 320)):
        """
        :param: size (default: tuple=(320, 320)) - target size
        """
        super().__init__()
        self.size=size
        
    def __call__(self, img, bbox=None):
        W, H = img.size
        img = img.resize(self.size)
        sW = self.size[0]/W
        sH = self.size[1]/H
        if bbox is None:
            return img
        bbox = [[int(b[0]*sW), int(b[1]*sH),
                 int(b[2]*sW), int(b[3]*sH), b[4]] for b in bbox]
        return img, bbox
    
class Rotate(Transformer):
    """Rotate image and bbox"""
    def __init__(self, p=0.5):
        """
        :param: p (default: float=0.5) - probability of rotation
        """
        super().__init__()
        self.p = p
        
    def __call__(self, img, bbox=None):
        if np.random.uniform() < self.p:
            img = img.rotate(90)
            W, H = img.size
            if bbox is not None:
                bbox = [[b[1], W - b[2], b[3], W - b[0], b[4]] for b in bbox]
        if bbox is None:
            return img
        return img, bbox
    
class HorizontalFlip(Transformer):
    """Horizontal flip of image and bbox"""
    def __init__(self, p=0.5):
        """
        :param: p (default: float=0.5) - probability of horizontal flip
        """
        super().__init__()
        self.p = p
        
    def __call__(self, img, bbox=None):
        if np.random.uniform() < self.p:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            W, H = img.size
            if bbox is not None:
                bbox = [[W - b[2], b[1], W - b[0], b[3], b[4]] for b in bbox]
        if bbox is None:
            return img
        return img, bbox
    
class VerticalFlip(Transformer):
    """Vertical flip of image and bbox"""
    def __init__(self, p=0.5):
        """
        :param: p (default: float=0.5) - probability of vertical flip
        """
        super().__init__()
        self.p = p
        
    def __call__(self, img, bbox=None):
        if np.random.uniform() < self.p:
            img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            W, H = img.size
            if bbox is not None:
                bbox = [[b[0], H - b[3], b[2], H - b[1], b[4]] for b in bbox]
        if bbox is None:
            return img
        return img, bbox
    
class ToTensor(Transformer):
    """Convert image and bbox to torch.tensor"""
    def __init__(self,):
        super().__init__()
    
    def __call__(self, img, bbox=None):
        img = np.array(img)/255
        img = torch.from_numpy(img).permute(2, 0, 1)
        if bbox is None:
            return img
        bbox = torch.tensor(bbox)
        return img, bbox
    
class ToImage(Transformer):
    """Convert image tensor to PIL.Image and bbox to list"""
    def __init__(self,):
        super().__init__()
    
    def __call__(self, img, bbox=None):
        img = img.permute(1,2,0).numpy()
        img = Image.fromarray(np.uint8(img*255))
        if bbox is None:
            return img
        bbox = bbox.tolist()
        return img, bbox
    
class Normalize(Transformer):
    """Normalize image and bbox"""
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        """
        :param: mean (default: list=[0.485, 0.456, 0.406]) - list of means for each image channel 
        :param: std (default: list=[0.229, 0.224, 0.225]) - list of stds for each image channel
        """
        super().__init__()
        self.mean=mean
        self.std=std
        
    def __call__(self, img, bbox=None):
        C, H, W = img.shape
        
        if (self.mean is not None) and (self.std is not None):
            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
            img = (img - mean)/std
            
        if bbox is None:
            return img
        scale = torch.tensor([[1/W, 1/H, 1/W, 1/H, 1]])
        bbox = bbox*scale
        return img, bbox
    
class DeNormalize(Transformer):
    """DeNormalize image and bbox"""
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        """
        :param: mean (default: list=[0.485, 0.456, 0.406]) - list of means for each image channel 
        :param: std (default: list=[0.229, 0.224, 0.225]) - list of stds for each image channel
        """
        super().__init__()
        self.mean=mean
        self.std=std
        
    def __call__(self, img, bbox=None):
        C, H, W = img.shape
        
        if (self.mean is not None) and (self.std is not None):
            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
            img = img*std + mean
            img = torch.clip(img, 0., 1.)
            
        if bbox is None:
            return img
        rescale = torch.tensor([[W, H, W, H, 1]])
        bbox = (bbox*rescale).long()
        return img, bbox
    
class PadBBox(Transformer):
    """Pad bboxes (to make it possible to get batches)"""
    def __init__(self, max_num_bbox=20, pad_value=-1.):
        """
        :param: max_num_bbox (default: int=20) - maximum number of possible bboxes
        :param: pad_value (default: float=0) - padding value
        """
        self.max_num_bbox=max_num_bbox
        self.pad_value=pad_value
        
    def __call__(self, img, bbox=None):
        if bbox is None:
            return img
        max_bbox = torch.full((self.max_num_bbox, 5), self.pad_value)
        num_bbox = len(bbox)
        if len(bbox) > self.max_num_bbox:
            bbox =  bbox[:self.max_num_bbox, :]
            num_bbox = self.max_num_bbox
        max_bbox[:num_bbox , :] = bbox
        return img, max_bbox
    
__all__ = ['Transformer', 'Compose', 'Resize', 'Rotate', 'HorizontalFlip',
           'VerticalFlip', 'ToTensor', 'ToImage', 'Normalize', 'DeNormalize', 'PadBBox']