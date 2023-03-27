import torch
import numpy as np
from PIL import Image
from abc import ABCMeta, abstractmethod
import cv2


def mutual_nn_matching_torch(desc1, desc2, threshold=None):
    if len(desc1) == 0 or len(desc2) == 0:
        return torch.empty((0, 2), dtype=torch.int64), torch.empty((0, 2), dtype=torch.int64)

    device = desc1.device
    desc1 = desc1 / desc1.norm(dim=1, keepdim=True)
    desc2 = desc2 / desc2.norm(dim=1, keepdim=True)
    similarity = torch.einsum('id, jd->ij', desc1, desc2)

    
    nn12 = similarity.max(dim=1)[1]
    nn21 = similarity.max(dim=0)[1]
    ids1 = torch.arange(0, similarity.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]]).t()
    scores = similarity.max(dim=1)[0][mask]    
    if threshold:
        mask = scores > threshold
        matches = matches[mask]    
        scores = scores[mask]
    return matches, scores

def mutual_nn_matching(desc1, desc2, threshold=None):
    if isinstance(desc1, np.ndarray):
        desc1 = torch.from_numpy(desc1)
        desc2 = torch.from_numpy(desc2)
    matches, scores = mutual_nn_matching_torch(desc1, desc2, threshold=threshold)
    return matches.cpu().numpy(), scores.cpu().numpy()

def resize_im(wo, ho, imsize=None, dfactor=1, value_to_scale=max):
    wt, ht = wo, ho
    if imsize and value_to_scale(wo, ho) > imsize and imsize > 0:
        scale = imsize / value_to_scale(wo, ho)
        ht, wt = int(round(ho * scale)), int(round(wo * scale))

    # Make sure new sizes are divisible by the given factor
    wt, ht = map(lambda x: int(x // dfactor * dfactor), [wt, ht])
    scale = (wo / wt, ho / ht)
    return wt, ht, scale

def read_im(im_path, imsize=None, dfactor=1):
    im = Image.open(im_path)
    im = im.convert('RGB')

    # Resize
    wo, ho = im.width, im.height
    wt, ht, scale = resize_im(wo, ho, imsize=imsize, dfactor=dfactor)
    im = im.resize((wt, ht), Image.BICUBIC)
    return im, scale

def read_im_gray(im_path, imsize=None):
    im, scale = read_im(im_path, imsize)
    return im.convert('L'), scale

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


class FeatureDetection(metaclass=ABCMeta):
    '''An abstract class for local feature detection and description methods'''
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
        torch.set_grad_enabled(False)
                
    @abstractmethod            
    def extract_features(self, im, **kwargs):
        """Given the processed input, the keypoints and descriptors are extracted by the model.
        Return:
            kpts : a Nx2 tensor, N is the number of keypoints.
            desc : a NxD tensor, N is the number of descriptors 
                   and D is dimension of each descriptor.            
        """
        
    @abstractmethod        
    def load_and_extract(self, im_path, **kwargs):
        """Given an image path, the input image is firstly loaded and processed accordingly,  
        the keypoints and descriptors are then extracted by the model.
        Return:
            kpts : a Nx2 tensor, N is the number of keypoints.
            desc : a NxD tensor, N is the number of descriptors 
                   and D is dimension of each descriptor.            
        """
                
    def describe(self, im, kpts, **kwargs):
        """Given the processed input and the pre-detected keypoint locations,
        feature descriptors are described by the model.
        Return:
            desc : a NxD tensor, N is the number of descriptors 
                   and D is dimension of each descriptor.
        """

    def detect(self, im, **kwargs):
        """Given the processed input, the keypoints are detected by that method.
        Return:
            kpts : a Nx2 tensor, N is the number of keypoints.
        """        
        
class Matching(metaclass=ABCMeta):
    '''An abstract class for a method that perform matching from the input pairs'''
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
        torch.set_grad_enabled(False)
    
    @classmethod
    def mutual_nn_match(self, desc1, desc2, threshold=0.0):
        """The feature descriptors from the pair of images are matched 
        using nearset neighbor search with mutual check and an optional 
        outlier filtering. This is normally used by feature detection methods.
        Args:
            desc1, desc2: descriptors from the 1st and 2nd image of a pair.
            threshold: the cosine similarity threshold for the outlier filtering.            
        Return:
            match_ids: the indices of the matched descriptors.
        """
        return mutual_nn_matching(desc1, desc2, threshold)
            
    @abstractmethod
    def match_pairs(self, im1_path, im2_path, **kwargs):
        """The model detects correspondences from a pair of images.
        All steps that are required to estimate the correspondences by a method
        are implemented here.
        Input:
            im1_path, im2_path: the paths of the input image pair.
            other args depend on the model.
            
        Return:
            matches: the detected matches stored as numpy array with shape Nx4,
                     N is the number of matches.
            kpts1, kpts2: the keypoints used for matching. For methods that don't 
                    explicitly define keypoints, e.g., SparseNCNet, 
                    the keypoints are the locations of points that get matched.
            scores: the matching score or confidence of each correspondence.
                    Notices, matching scores are defined differently across methods.
                    For NN matcher, they can be the cosine distance of descriptors;
                    For SuperGlue, they are the probablities in the OT matrix; ..etc.                    
        """
    
