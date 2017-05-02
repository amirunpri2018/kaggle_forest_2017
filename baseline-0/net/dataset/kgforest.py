from net.common import *
from net.dataset.tool import *
import pandas as pd
from skimage import io

from net.dataset.standard import *

# helper functions
def draw_multi_classes(image, classes, label, threshold, name=''):
    draw_shadow_text(image, name, (5,15),  0.5, (255,255,255), 1)

    s = get_multi_classes(label, classes, threshold, nil='nil')
    for i,ss in enumerate(s.split(' ')):
        draw_shadow_text(image, ' '+ss, (5,30+i*15),  0.5, (255,255,255), 1)


## custom data loader -----------------------------------
class KgForestDataset(Dataset):
    DATA_DIR = '/root/share/data/kaggle-forest'

    def __init__(self, split, transform=None, ext='jpg', height=64, width=64):
        data_dir = self.DATA_DIR

        # read classes
        with open(data_dir + '/classes') as f:
            classes = f.readlines()
        classes = [x.strip() for x in classes]

        # read all annotations
        csv_file  = self.DATA_DIR + '/image/train_label.csv'
        df = pd.read_csv(csv_file)
        for c in classes:
            df[c] = df['tags'].apply(lambda x: 1 if c in x.split(' ') else 0)
        # df = df.sort_values(by=['image_name'], ascending=True)


        # read images and labels
        list = data_dir +'/split/'+ split
        with open(list) as f:
            names = f.readlines()
        names = [x.strip() for x in names]

        num_classes = len(classes)
        N = len(names)
        images = np.zeros((N,height,width,3),dtype=np.uint8)
        labels = np.zeros((N,num_classes),dtype=np.float32)

        df1=df.set_index('image_name')
        if ext=='jpg':
            for n in range(N):
                shortname = names[n].split('/')[-1].replace('.jpg','')
                image = cv2.imread(data_dir + '/image/' + names[n],1)
                h,w=image.shape[0:2]

                if height!=h or width!=w:
                    image=cv2.resize(image,(height,width))

                images[n] = image
                labels[n] = df1.loc[shortname].values[1:]

                if 1: #debug
                    #draw_shadow_text(images[n], shortname, (5,15),  0.5, (255,255,255), 1)
                    #draw_multi_classes(images[n],classes,labels[n],0.5,shortname)
                    #im_show('image', images[n], resize=1 )
                    #cv2.waitKey(0)
                    #print('cv2 read: %s'%name)
                    pass
        elif ext=='tif':
            ## <todo>
            pass

        self.df=df
        self.transform=transform
        self.classes = classes
        self.names   = names
        self.images  = images
        self.labels  = labels #t = torch.from_numpy(a)
        self.num = len(images)


    #https://discuss.pytorch.org/t/trying-to-iterate-through-my-custom-dataset/1909
    def __getitem__(self, index):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)
        img   = self.images[index]
        label = self.labels[index]

        img = PIL.Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, index


    def __len__(self):
        #print ('\tcalling Dataset:__len__')
        return len(self.images)




def check_kgforest_dataset(dataset, loader, height=256, width=256):

    classes = dataset.classes
    names   = dataset.names

    for i, (images, labels, indices) in enumerate(loader, 0):
        print('i=%d: '%(i))

        # get the inputs
        num = len(images)
        for n in range(num):
            label=labels[n].numpy()
            s = get_multi_classes(label, classes)
            print('%32s : %s %s'%(names[indices[n]], label.T, s))

            image = tensor_to_img(images[n]).copy()
            h,w = image.shape[0:2]
            if height!=h or width!=w:
                image=cv2.resize(image,(height,width))

            shortname = names[indices[n]].split('/')[-1].replace('.jpg','')
            draw_multi_classes(image, classes, label, 0.5, shortname)

            im_show('image',image , resize=1 )
            cv2.waitKey(0)
            #print('\t\tlabel=%d : %s'%(label,classes[label]))
            #print('')



## other functions  -------------
def run_split_list():
    list='/root/share/data/kaggle-forest/split/train-40479'
    train_list='/root/share/data/kaggle-forest/split/train-39479'
    valid_list='/root/share/data/kaggle-forest/split/valid-1000'

    num=40479
    valid_num=1000
    train_num=num-valid_num

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



def run_make_cvs():
    dataset = KgForestDataset('valid-1000', transform=None) #train-39479  #valid-1000  #train-sample
    csv_file='/root/share/data/kaggle-forest/image/valid-1000.csv'

    with open(csv_file,'w') as f:
        f.write('image_name,tags\n')
        num= dataset.num
        for n in range(num):
            name = dataset.names[n]
            shortname = name.split('/')[-1].replace('.jpg','')
            s = get_multi_classes(dataset.labels[n], dataset.classes, threshold=-0)
            f.write('%s,%s\n'%(shortname,s))

            pass
        pass



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    if 1:
        ##run_split_list()
        ##run_make_classes()
        ##run_make_cvs()
        exit(0)


    transform = transforms.Compose([
                    # transforms.ToTensor(): Converts a PIL.Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: dummy_transform1(x)),
                    transforms.Lambda(lambda x: dummy_transform2(x)),
                ])

    dataset = KgForestDataset('train-order-10', transform=transform) #train-39479  #valid-1000  #train-sample
    sampler = SequentialSampler(dataset)  #SequentialSampler  #RandomSampler
    loader  = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2, drop_last=False, pin_memory=True)



    names = dataset.df['image_name'].values
    for epoch in range(10):
        print('epoch=%d -------------------------'%(epoch))
        #check_kgforest_dataset(dataset, loader)

        #debug using iter + next()
        classes = dataset.classes
        names   = dataset.names

        iterator  = iter(loader)
        num_iters = len(loader)
        for i in range(num_iters):

            (images, labels, indices) = iterator.next()
            num = len(images)

            print('i=%d/%d (%d): '%(i,num_iters,num))
            for n in range(num):
                label=labels[n].numpy()
                s = get_multi_classes(label, classes)
                print('%32s : %s %s'%(names[indices[n]], label.T, s))

                image = tensor_to_img(images[n]).copy()
                image = cv2.resize(image,(256,256),interpolation=cv2.INTER_NEAREST)
                shortname = names[indices[n]].split('/')[-1].replace('.jpg','')
                if 1: #debug
                    draw_multi_classes(image, classes, label, 0.5, shortname)
                    pass

                im_show('image',image , resize=1 )
                cv2.waitKey(0)
                #print('\t\tlabel=%d : %s'%(label,classes[label]))
                #print('')

    print('sucess')