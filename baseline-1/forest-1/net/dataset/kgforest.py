from net.common import *
from net.dataset.tool import *
import pandas as pd

from skimage import io


# helper functions -------------
def get_multi_classes(score, classes, threshold = 0.5, nil=''):

    N = len(classes)
    if not isinstance(threshold,list) : threshold = [threshold]*N

    s=nil
    for n in range(N):
        if score[n]>threshold[n]:
            if s==nil:
                s = classes[n]
            else:
                s = '%s %s'%(s,classes[n])
    return s



def draw_multi_classes(image, classes, label, threshold=0.5):

    weather_classes = classes[:4]
    other_classes   = classes[4:]
    weather_label = label[:4]
    other_label   = label[4:]

    ss = weather_classes[np.argmax(weather_label)]
    draw_shadow_text(image, ' '+ss, (5,30),  0.5, (255,255,0), 1)

    s = get_multi_classes(other_label, other_classes, threshold, nil=' ')
    for i,ss in enumerate(s.split(' ')):
        draw_shadow_text(image, ' '+ss, (5,30+(i+1 )*15),  0.5, (0,255,255), 1)



## custom data tarnsform  -----------------------------------

def tensor_to_img(img, mean=0, std=1, dtype=np.uint8):
    img = np.transpose(img.numpy(), (1, 2, 0))
    img = (img*std+ mean)
    img = img.astype(dtype)
    #img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return img

## transform (input is numpy array, read in by cv2)
def toTensor(img):
    img = img.transpose((2,0,1)).astype(np.float32)
    tensor = torch.from_numpy(img).float()
    return tensor


#http://enthusiaststudent.blogspot.jp/2015/01/horizontal-and-vertical-flip-using.html
#http://qiita.com/supersaiakujin/items/3a2ac4f2b05de584cb11
def randomVerticalFlip(img, u=0.5):
    if random.random() < u:
        img = cv2.flip(img,0)  #np.flipud(img)  #cv2.flip(img,0) ##up-down
    return img

def randomHorizontalFlip(img, u=0.5):
    if random.random() < u:
        img = cv2.flip(img,1)  #np.fliplr(img)  #cv2.flip(img,1) ##left-right
    return img


def randomFlip(img, u=0.5):
    if random.random() < u:
        img = cv2.flip(img,random.randint(-1,1))
    return img

def randomTranspose(img, u=0.5):
    if random.random() < u:
        img = cv2.transpose(img)
    return img

#http://stackoverflow.com/questions/16265673/rotate-image-by-90-180-or-270-degrees
def randomRotate90(img, u=0.25):
    if random.random() < u:
        angle=random.randint(1,3)*90
        if angle == 90:
            img = cv2.transpose(img)
            img = cv2.flip(img,1)
            #return img.transpose((1,0, 2))[:,::-1,:]
        elif angle == 180:
            img = cv2.flip(img,-1)
            #return img[::-1,::-1,:]
        elif angle == 270:
            img = cv2.transpose(img)
            img = cv2.flip(img,0)
            #return  img.transpose((1,0, 2))[::-1,:,:]
    return img




## custom data sampler  -----------------------------------
# see torch/utils/data/sampler.py
#<todo> change to balance class sampler .... currently, just a random sampler
class KgForestSampler(Sampler):
    def __init__(self, data):
        self.num_samples = len(data)

    def __iter__(self):
        #print ('\tcalling Sampler:__iter__')
        return iter(torch.randperm(self.num_samples).long())

    def __len__(self):
        #print ('\tcalling Sampler:__len__')
        return self.num_samples




## custom data loader -----------------------------------
class KgForestDataset(Dataset):

    def __init__(self, split, transform=None, ext='jpg', height=64, width=64, label_csv='train_label.csv'):
        data_dir = '/root/share/data/kaggle-forest'

        # read classes
        with open(data_dir + '/classes') as f:
            classes = f.readlines()
        classes = [x.strip() for x in classes]
        num_classes = len(classes)


        # read names
        list = data_dir +'/split/'+ split
        with open(list) as f:
            names = f.readlines()
        names = [x.strip().replace('<ext>',ext) for x in names]
        num   = len(names)


        #read images
        images = None
        if ext=='jpg':
            images = np.zeros((num,height,width,3),dtype=np.uint8)
            for n in range(num):
                img_file=data_dir + '/image/' + names[n]
                image = cv2.imread(img_file,1)
                h,w=image.shape[0:2]
                if height!=h or width!=w:
                    image=cv2.resize(image,(height,width))
                images[n] = image

                if 0 : #debug
                    resize = (h/height + w/width)/2
                    im_show('image',image, resize)
                    cv2.waitKey(0)
                #cv2.circle(images[n],(0,0),10,(255,255,255),-1)

        elif ext=='tif':
            #https://blog.philippklaus.de/2011/08/handle-16bit-tiff-images-in-python
            images = np.zeros((num,height,width,4),dtype=np.uint16)
            for n in range(num):
                img_file=data_dir + '/image/' + names[n]

                #image = cv2.imread(img_file,-1)
                image = io.imread(img_file)
                h,w = image.shape[0:2]
                if height!=h or width!=w:
                    image = cv2.resize(image,(height,width))
                images[n] = image

                if 0 : #debug
                    resize = (h/height + w/width)/2
                    image   = cv2.convertScaleAbs(image, alpha=(255.0/65535.0)*4)
                    img_rgb = image[:,:,0:3]
                    img_n   = image[:,:,3]
                    im_show('rgb',img_rgb,resize)
                    im_show('n',img_n,resize)
                    im_show('image',image,resize)
                    cv2.waitKey(0)
                #cv2.circle(images[n],(0,0),10,(255,255,255,255),-1)
            pass
        else:
            raise ValueError('KgForestDataset: unsupported ext = %s'%ext)


        #read labels
        labels = None
        df     = None
        if label_csv is not None:
            labels = np.zeros((num,num_classes),dtype=np.float32)

            csv_file  = data_dir + '/image/' + label_csv   # read all annotations
            df = pd.read_csv(csv_file)
            for c in classes:
                df[c] = df['tags'].apply(lambda x: 1 if c in x.split(' ') else 0)
            # df = df.sort_values(by=['image_name'], ascending=True)

            df1 = df.set_index('image_name')
            for n in range(num):
                shortname = names[n].split('/')[-1].replace('.'+ext,'')
                labels[n] = df1.loc[shortname].values[1:]

                if 0: #debug
                    image = cv2.resize(images[n],(256,256),interpolation=cv2.INTER_NEAREST)
                    draw_shadow_text  (image, shortname, (5,15),  0.5, (255,255,255), 1)
                    draw_multi_classes(image, classes, labels[n])
                    im_show('image', image)
                    cv2.waitKey(0)

                    #images[n]=cv2.resize(image,(height,width)) ##mark for debug
                    pass
        #save
        self.transform = transform
        self.ext       = ext
        self.num       = num
        self.names     = names
        self.images    = images


        self.classes = classes
        self.df      = df
        self.labels  = labels


    #https://discuss.pytorch.org/t/trying-to-iterate-through-my-custom-dataset/1909
    def __getitem__(self, index):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)

        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.labels is None:
            return img, index

        else:
            label = self.labels[index]
            return img, label, index


    def __len__(self):
        #print ('\tcalling Dataset:__len__')
        return len(self.images)




def check_kgforest_dataset(dataset, loader):
    height,width = 256,256

    classes = dataset.classes
    names   = dataset.names
    ext     = dataset.ext

    if dataset.labels is not None:
        for i, (images, labels, indices) in enumerate(loader, 0):
            print('i=%d: '%(i))

            # get the inputs
            num = len(images)
            for n in range(num):
                label = labels[n].numpy()
                s = get_multi_classes(label, classes)
                print('%32s : %s %s'% \
                      (names[indices[n]], label.T, s))

                if ext=='tif':
                    image = tensor_to_img(images[n],dtype=np.uint16).copy()
                    image  = cv2.convertScaleAbs(image, alpha=(255.0/65535.0)*4)
                if ext=='jpg':
                    image = tensor_to_img(images[n],dtype=np.uint8).copy()

                h,w   = image.shape[0:2]
                if height!=h or width!=w:
                    image=cv2.resize(image,(height,width))


                shortname = names[indices[n]].split('/')[-1].replace('.'+ext,'')
                draw_shadow_text  (image, shortname, (5,15),  0.5, (255,255,255), 1)
                draw_multi_classes(image, classes, label)
                im_show('image',image)
                cv2.waitKey(0)
                #print('\t\tlabel=%d : %s'%(label,classes[label]))
                #print('')

    if dataset.labels is None:
        for i, (images, indices) in enumerate(loader, 0):
            print('i=%d: '%(i))

            # get the inputs
            num = len(images)
            for n in range(num):
                print('%32s : nil'% (names[indices[n]]))

                if ext=='tif':
                    image = tensor_to_img(images[n],dtype=np.uint16).copy()
                    image  = cv2.convertScaleAbs(image, alpha=(255.0/65535.0)*4)
                if ext=='jpg':
                    image = tensor_to_img(images[n],dtype=np.uint8).copy()

                h,w = image.shape[0:2]
                if height!=h or width!=w:
                    image=cv2.resize(image,(height,width))

                shortname = names[indices[n]].split('/')[-1].replace('.'+ext,'')
                draw_shadow_text  (image, shortname, (5,15),  0.5, (255,255,255), 1)
                im_show('image',image)
                cv2.waitKey(0)





## other run functions  -------------
def run_split_list():
    SEED=12345
    list='/root/share/data/kaggle-forest/split/train-40479'

    num=40479
    valid_num=3000
    train_num=num-valid_num
    train_list='/root/share/data/kaggle-forest/split/train-%d'%train_num
    valid_list='/root/share/data/kaggle-forest/split/valid-%d'%valid_num

    with open(list) as f:
        names = f.readlines()
    names = [x.strip() for x in names]
    names.sort()
    random.shuffle(names)

    train_names= names[0:train_num]
    with open(train_list,'w') as f:
        for name in train_names:
          f.write('%s\n' % name)

    valid_names = names[train_num:]
    with open(valid_list,'w') as f:
        for name in valid_names:
          f.write('%s\n' % name)


def run_make_classes():

    csv_file      = '/root/share/data/kaggle-forest/image/train_label.csv'
    classes_file  = '/root/share/data/kaggle-forest/classes'
    df = pd.read_csv(csv_file)

    # build list with unique labels
    classes = []
    for value in df.tags.values:
        cs = value.split(' ')
        for c in cs:
            if c not in classes:
                classes.append(c)
    classes.sort()

    with open(classes_file,'w') as f:
        for c in classes:
          f.write('%s\n' % c)




def run_find_statistics():
    data_dir = '/root/share/data/kaggle-forest'
    split = 'train-1000'

    # read names
    list = data_dir +'/split/'+ split
    with open(list) as f:
        names = f.readlines()
    names = [x.strip() for x in names]
    num = len(names)

    width  = 256
    height = 256
    all = np.zeros((4,num,height*width), np.uint16)
    for n in range(num):
        img_file=data_dir + '/image/' + names[n]
        img_file = img_file.replace('jpg', 'tif')

        print ('\r%8d/%d: %s'%(n,num,img_file),end='',flush=True)
        #img = cv2.imread(img_file,-1)
        img = io.imread(img_file)
        img = img.reshape((-1,4))

        all[0,n] = img[:,0]
        all[1,n] = img[:,1]
        all[2,n] = img[:,2]
        all[3,n] = img[:,3]
    print ('')

    all   = np.array(all, dtype=np.float32).reshape(4,-1)
    means = np.mean(all, axis=1)
    stds  = np.std (all, axis=1)

    print('means=%s, stds=%s, num=%d'%(means, stds, num))




def run_check_files():
    data_dir = '/root/share/data/kaggle-forest'


    # num = 40479
    # for n in range(num):
    #     img_file=data_dir + '/image/train-tif/train_%d.tif'%n
    #
    num = 40669
    for n in range(num):
        img_file=data_dir + '/image/test-tif/test_%d.tif'%n

        try:
            img = io.imread(img_file)
            # print(img)
            # print(img.shape)
            # print(type(img))
        except:
            print('error: %s'%img_file)
        pass




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    if 0:
        run_split_list()
        ##run_make_classes()
        ##run_make_cvs()
        #run_find_statistics()

        #run_check_files()
        exit(0)


    dataset = KgForestDataset('train-ordered-20', ##'train-40479',  ##'train-ordered-20', ##
                                transforms.Compose([
                                    #transforms.Lambda(lambda x: randomVerticalFlip(x)),
                                    #transforms.Lambda(lambda x: randomHorizontalFlip(x)),
                                    #transforms.Lambda(lambda x: randomRotate90(x)),
                                    transforms.Lambda(lambda x: randomFlip(x)),
                                    transforms.Lambda(lambda x: randomTranspose(x)),
                                    transforms.Lambda(lambda x: toTensor(x)),
                                ]),
                                ext='jpg',  # ext='tif', #
                                #label_csv=None,
                              )
    sampler = KgForestSampler(dataset)  #SequentialSampler  #RandomSampler
    loader  = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2, drop_last=False, pin_memory=True)


    for epoch in range(10):
        print('epoch=%d -------------------------'%(epoch))
        check_kgforest_dataset(dataset, loader)

    print('sucess')