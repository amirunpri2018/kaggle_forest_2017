from net.common import *
from net.dataset.tool import *



## custom data loader -----------------------------------
# see torchvision/dataset/cifar.py
class CifarDataset(Dataset):
    DATA_DIR = '/root/share/project/pytorch/data/cifar10/rgb'
    WIDTH =32
    HEIGHT=32

    def __init__(self, split, transform=None, data_dir=None):
        data_dir = self.DATA_DIR if data_dir is None else data_dir

        list = data_dir +'/split/'+ split
        with open(list) as f:
            names = f.readlines()
        names = [x.strip() for x in names]

        # load images and labels
        N = len(names)
        images = np.zeros((N,self.HEIGHT,self.WIDTH,3),dtype=np.uint8)
        labels = []  #np.zeros((N),dtype=np.int32)

        for n in range(N):
            images[n] = cv2.imread(data_dir + '/image/' + names[n],1)
            s = names[n].split('/')
            labels += [int(s[-2])]


        self.transform=transform
        self.classes=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.images=images
        self.labels=labels
        self.num=len(images)


    def __getitem__(self, index):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)
        img   = self.images[index]
        label = self.labels[index]

        #img = PIL.Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, index


    def __len__(self):
        #print ('\tcalling Dataset:__len__')
        return len(self.images)







# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    transform = transforms.Compose([
                    # transforms.ToTensor(): Converts a PIL.Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
                    transforms.ToTensor(),
                    #transforms.Lambda(lambda x: dummy_transform1(x)),
                    #transforms.Lambda(lambda x: dummy_transform2(x)),
                ])

    dataset = CifarDataset('train10', transform=transform)
    sampler = SequentialSampler(dataset)  #SequentialSampler  #RandomSampler
    loader  = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2, drop_last=True)  ##shuffle=True #False
                         # shuffle=False
                         # pin_memory=True
                         # drop_last=False


    classes = dataset.classes
    for epoch in range(10):
        print('epoch=%d -------------------------'%(epoch))
        for i, (images, labels, indices) in enumerate(loader, 0):
            print('i=%d: '%(i))

            num = len(images)
            for n in range(num):
                image=images[n]
                im_show('image', tensor_to_img(image), resize=6 )
                cv2.waitKey(1)
                label=labels[n]
                print('\t\tlabel=%d : %s'%(label,classes[label]))
                #print('')

    dd=0
    print('sucess')