# explore dataset: https://www.kaggle.com/robinkraft/planet-understanding-the-amazon-from-space/getting-started-with-the-data-now-with-docs
#                  https://www.kaggle.com/robinkraft/planet-understanding-the-amazon-from-space/issue-with-tif-files

from net.common import *
from net.dataset.tool import *
import pandas as pd
from skimage import io


#  <todo> try different normlisation methods ...
#
#
#

DATA_DIR = '/root/share/data/kaggle-forest'
MEANS=[4894.064,  4153.9741, 2978.3396, 6164.8926]
STDS =[1972.3749, 1575.0253, 1735.2235, 1973.5652]


# compute mean and std
def get_statistics(files):

    all = [[],[],[],[]]
    for img_file in files:
        print (img_file)
        img = io.imread(img_file)
        img = img.reshape((-1,4))
        all[0] = all[0] + img[:,0].tolist()
        all[1] = all[1] + img[:,1].tolist()
        all[2] = all[2] + img[:,2].tolist()
        all[3] = all[3] + img[:,3].tolist()

    all = np.array(all, dtype=np.float32)
    means = np.mean(all,axis=0)
    stds  = np.std(all,axis=0)
    count = len(files)

    return means, stds, count




def run_xxx_0():
    csv_file = DATA_DIR + '/image/train_label.csv'
    labels_df = pd.read_csv(csv_file)

    # Build list with unique labels
    classes = []
    for value in labels_df.tags.values:
        cs = value.split(' ')
        for c in cs:
            if c not in classes:
                classes.append(c)
    classes.sort()

    print(classes)
    print(len(classes))

    for c in classes:
        labels_df[c] = labels_df['tags'].apply(lambda x: 1 if c in x.split(' ') else 0)

    # Display head
    print(labels_df.head())

    ##load an image
    name='train_1'
    img_file = DATA_DIR + '/image/train-tif-sample/' + name + '.tif'

    img = io.imread(img_file)

    #normalise to [0, 255]
    files = sorted(glob.glob(DATA_DIR + '/image/train-tif-sample' +'/*.tif'))
    #np.random.shuffle(files)
    if 0:
        means, stds, count = get_statistics(files)
    else:
        means = np.array(MEANS)
        stds  = np.array(STDS )
        count = 0

    print(means, stds, count)



    os.makedirs(DATA_DIR + '/image/train-tif-sample_rgb',exist_ok=True)
    os.makedirs(DATA_DIR + '/image/train-tif-sample_n',  exist_ok=True)
    for img_file in files:
        name = os.path.basename(img_file).replace('.tif','')

        print (img_file)
        img = io.imread(img_file)
        img = img.astype(np.float32)

        img_flat = img.reshape(-1,4)
        #means = np.mean(img_flat,axis=0)
        #stds  = np.std (img_flat,axis=0)

        img = (img-means)/stds
        img = np.clip((img+1)*128,0,255).astype(np.uint8)

        img_rgb = img[:,:,0:3]
        img_n   = img[:,:,3]

        cv2.imwrite(DATA_DIR + '/image/train-tif-sample_rgb/'+name+'.png',img_rgb)
        cv2.imwrite(DATA_DIR + '/image/train-tif-sample_n/'+name+'.png',img_n)

        im_show('rgb',img_rgb,resize=2)
        im_show('n',img_n,resize=2)
        cv2.waitKey(1)

    pass


def run_xxx():

    ##load an image
    name='train_1'
    img_file = DATA_DIR + '/image/train-tif-sample/' + name + '.tif'

    img = io.imread(img_file)

    #normalise to [0, 255]
    files = sorted(glob.glob(DATA_DIR + '/image/train-tif-sample' +'/*.tif'))
    #np.random.shuffle(files)

    if 0:
        means, stds, count = get_statistics(files)
    else:
        means = np.array(MEANS)
        stds  = np.array(STDS )
        count = 0

    print(means, stds, count)



    os.makedirs(DATA_DIR + '/image/train-tif-sample_rgb',exist_ok=True)
    os.makedirs(DATA_DIR + '/image/train-tif-sample_n',  exist_ok=True)
    for img_file in files:
        name = os.path.basename(img_file).replace('.tif','')

        print (img_file)
        img = io.imread(img_file)
        img = img.astype(np.float32)

        img_flat = img.reshape(-1,4)
        #means = np.mean(img_flat,axis=0)
        #stds  = np.std (img_flat,axis=0)

        img = (img-means)/stds
        img = np.clip((img+1)*128,0,255).astype(np.uint8)

        img_rgb = img[:,:,0:3]
        img_n   = img[:,:,3]

        cv2.imwrite(DATA_DIR + '/image/train-tif-sample_rgb/'+name+'.png',img_rgb)
        cv2.imwrite(DATA_DIR + '/image/train-tif-sample_n/'+name+'.png',img_n)

        im_show('rgb',img_rgb,resize=2)
        im_show('n',img_n,resize=2)
        cv2.waitKey(1)

    pass
# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))



    run_xxx()