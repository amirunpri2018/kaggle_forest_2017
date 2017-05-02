from net.common import *


# common tool for dataset

# draw -----------------------------------
def im_show(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))


def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)




# data -----------------------------------
def tensor_to_img(img, mean=0, std=1):
    img = np.transpose(img.numpy(), (1, 2, 0))
    img = (img*std+ mean)*255
    img = img.astype(np.uint8)
    #img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return img


## Augmentation code here -------------------
# geometric transform
# illumination transform
# etc

def dummy_transform1(image):
    print ('\t\tdummy_transform1')
    return image
def dummy_transform2(image):
    print ('\t\tdummy_transform2')
    return image

## Sampler code here -----------------------------------

# see trorch/utils/data/sampler.py
class DummySampler(Sampler):
    def __init__(self, data):
        self.num_samples = len(data)

    def __iter__(self):
        #print ('\tcalling Sampler:__iter__')
        l = list(range(self.num_samples))
        #random.shuffle(l)
        return iter(l)

    def __len__(self):
        #print ('\tcalling Sampler:__len__')
        return self.num_samples
