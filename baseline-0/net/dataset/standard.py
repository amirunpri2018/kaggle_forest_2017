from net.common import *
from net.dataset.tool import *

from skimage import io


# helper functions
def get_multi_classes(score, classes, threshold=0.5, nil=''):

    s=nil
    N=len(classes)
    for n in range(N):
        if score[n]>threshold:
            if s==nil:
                s = classes[n]
            else:
                s = '%s %s'%(s,classes[n])
    return s


#fixed size images ---------------------------------------
class CropDataset(Dataset):

    def __init__(self, split, data_dir, transform=None, height=64, width=64):

        # read classes
        with open(data_dir + '/classes') as f:
            classes = f.readlines()
        classes = [x.strip() for x in classes]


        # read images
        list = data_dir +'/split/'+ split
        with open(list) as f:
            names = f.readlines()
        names = [x.strip() for x in names]


        N = len(names)
        images = np.zeros((N, width, height, 3),dtype=np.uint8)
        for n in range(N):
            image = cv2.imread(data_dir + '/image/' + names[n],1)
            h,w = image.shape[0:2]
            if height!=h or width!=w:
                image=cv2.resize(image,(height,width))

            images[n] = image

        self.transform = transform
        self.classes   = classes
        self.names     = names
        self.images    = images
        self.num       = len(images)


    def __getitem__(self, index):
        img  = self.images[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, index

    def __len__(self):
        return len(self.images)




def check_crop_dataset(dataset, loader, height=256, width=256):

    classes = dataset.classes
    names   = dataset.names
    for i, (images, indices) in enumerate(loader, 0):
        print('i=%d: '%(i))

        # get the inputs
        num = len(images)
        for n in range(num):
            print('%32s :'%(names[indices[n]]))

            image = tensor_to_img(images[n]).copy()
            h,w = image.shape[0:2]
            if height!=h or width!=w:
                image=cv2.resize(image,(height,width))

            shortname = names[indices[n]].split('/')[-1].replace('.jpg','')
            im_show('image',image , resize=1 )
            cv2.waitKey(0)

