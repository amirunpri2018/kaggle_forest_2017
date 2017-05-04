from net.common import *
from net.dataset.tool import *
from net.utility.tool import *

from net.dataset.kgforest import *

##from net.model.vggnet import VggNet as Net
from net.model.simplenet import SimpleNet64_2 as Net
##from net.model.resnet import ResNet18 as Net

from sklearn.metrics import fbeta_score


## global setting -------------
EXT  = 'jpg'  #'tif'  #'jpg'  #
SIZE = 64

##-----------------------------


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr


## F measure : https://www.kaggle.com/c/planet-understanding-the-amazon-from-space#evaluation
#    f0 = fbeta_score(labels, probs, beta=2, average='micro')
def logits_to_probs(logits, is_force_single_weather=False):

    probs = F.sigmoid(Variable(logits)).data
    if is_force_single_weather:
        weather = logits[:,0:4]
        maxs, indices = torch.max(weather, 1)
        weather.zero_()
        weather.scatter_(1, indices, 1)
        probs[:,0:4] = weather
        #print(probs)

    return probs


def f_measure( logits, labels, threshold=0.23, beta=2 ):

    SMALL = 1e-6 #0  #1e-12
    batch_size = logits.size()[0]

    #weather
    probs  = logits_to_probs(logits)
    l = labels
    p = (probs>threshold).float()

    num_pos     = torch.sum(p,  1) + SMALL
    num_pos_hat = torch.sum(l,  1)
    tp          = torch.sum(l*p,1)
    precise     = tp/num_pos
    recall      = tp/num_pos_hat

    fs = (1+beta*beta)*precise*recall/(beta*beta*precise + recall + SMALL)
    f  = fs.sum()/batch_size
    return f





#https://www.kaggle.com/paulorzp/planet-understanding-the-amazon-from-space/find-best-f2-score-threshold/code
def find_f_measure_threshold(probs, labels, thresholds=None):

    #f0 = fbeta_score(labels, probs, beta=2, average='samples')  #micro  #samples
    def _f_measure(probs, labels, threshold=0.5, beta=2 ):

        SMALL = 1e-12 #0  #1e-12
        batch_size, num_classes = labels.shape[0:2]

        l = labels
        p = probs>threshold

        num_pos     = p.sum(axis=1) + SMALL
        num_pos_hat = l.sum(axis=1)
        tp          = (l*p).sum(axis=1)
        precise     = tp/num_pos
        recall      = tp/num_pos_hat

        fs = (1+beta*beta)*precise*recall/(beta*beta*precise + recall + SMALL)
        f  = fs.sum()/batch_size
        return f


    best_threshold =  0
    best_score     = -1

    if thresholds is None:
        thresholds = np.arange(0,1,0.005)
        ##thresholds = np.unique(probs)

    N=len(thresholds)
    scores = np.zeros(N,np.float32)
    for n in range(N):
        t = thresholds[n]
        #score = f_measure(probs, labels, threshold=t)
        score = fbeta_score(labels, probs>t, beta=2, average='samples')  #micro  #samples
        scores[n] = score

    return thresholds,scores


## https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
def find_f_measure_threshold2(probs, labels, resolution=100):
    p,y = probs, labels
    batch_size, num_classes = labels.shape[0:2]

    def mf(x):
        p2 = np.zeros_like(p)
        for i in range(17):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)
        score = fbeta_score(y, p2, beta=2, average='samples')
        return score

    thesholds = [0.2]*num_classes
    scores = [0]*num_classes
    for i in range(num_classes):

        best_theshold = 0
        best_score    = 0
        for theshold in range(1,resolution):
            theshold /= resolution
            thesholds[i] = theshold
            score = mf(thesholds)
            if score > best_score:
                best_theshold = theshold
                best_score = score

        thesholds[i] = best_theshold
        scores[i]    = best_score
        print('\t(i, best_theshold, best_score)=%2d, %0.3f, %f'%(i, best_theshold, best_score))

    print('')
    return thesholds, scores


def plot_f_measure_threshold(thresholds, scores):

    plt.plot(thresholds, scores)
    plt.grid(True)
    plt.yticks(np.arange(0.7,  1, 0.05))
    plt.xticks(np.arange(0,  0.5, 0.1))
    plt.ylim(0.7,1)
    plt.xlim(0,0.5)



# loss ----------------------------------------
def criterion(logits, labels):
    loss = nn.MultiLabelSoftMarginLoss()(logits, Variable(labels))
    return loss


def evaluate(net, test_loader):

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (images, labels, indices) in enumerate(test_loader, 0):

        # forward
        logits = net(Variable(images.cuda()))
        loss   = criterion(logits, labels.cuda())

        batch_size = len(images)
        test_acc  += batch_size*f_measure(logits.data, labels.cuda())
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num

    return test_loss, test_acc, test_num





## main functions ############################################################3
def do_training():

    out_dir ='/root/share/project/pytorch/results/kaggle-forest/new-xx05'
    initial_model = None   #None '/root/share/project/pytorch/results/kaggle-forest/xx11/snap/final.torch'

    os.makedirs(out_dir +'/snap', exist_ok=True)
    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED    = %u\n' % SEED)
    log.write('\tfile    = %s\n' % __file__)
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    batch_size    = 96 #128 #128
    height, width = SIZE, SIZE

    train_dataset = KgForestDataset( 'train-37479',  #'train-1000',   #
                                    transform=transforms.Compose([
                                        transforms.Lambda(lambda x: randomFlip(x)),
                                        transforms.Lambda(lambda x: randomTranspose(x)),
                                        transforms.Lambda(lambda x: toTensor(x)),
                                    ]),
                                    ext=EXT,
                                    height=height,width=width,label_csv='train_label.csv')
    train_loader  = DataLoader(
                        train_dataset,
                        sampler=KgForestSampler(train_dataset),
                        batch_size=batch_size,
                        drop_last=True,
                        num_workers=5,
                        pin_memory=True)

    test_dataset = KgForestDataset('valid-3000', #'valid-1000',  #
                                   transform=transforms.Compose([
                                         transforms.Lambda(lambda x: toTensor(x)),
                                   ]),
                                   ext=EXT,
                                   height=height,width=width,label_csv='train_label.csv')
    test_loader  = DataLoader(
                        test_dataset,
                        sampler=None,
                        batch_size=batch_size,
                        drop_last=False,
                        num_workers=5,
                        pin_memory=True)

    in_channels = train_dataset.images.shape[3]
    num_classes = 17

    log.write('\t(height,width)    = (%d, %d)\n'%(height,width))
    log.write('\tin_channels       = %d\n'%(in_channels))
    log.write('\tEXT               = %s\n'%(EXT))
    log.write('\ttrain_dataset.num = %d\n'%(train_dataset.num))
    log.write('\ttest_dataset.num  = %d\n'%(test_dataset.num))
    log.write('\tbatch_size = %d\n'%batch_size)
    log.write('\n')

    if 0:  ## check data
        check_kgforest_dataset(train_dataset, train_loader)
        exit(0)

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net((in_channels,height,width), num_classes)
    if initial_model is not None:
        net = torch.load(initial_model)

    net.cuda()
    log.write('\n%s\n'%(str(net)))
    log.write('\n')


    ## optimiser ----------------------------------
    num_epoches=70
    it_print  =1
    epoch_test=1
    epoch_save=1

    #https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/4
    optimizer = optim.SGD(net.parameters(), lr=0, momentum=0.9, weight_decay=0.0005)
    #optimizer = optim.Adam(net.parameters(), lr=1e-3)  #, weight_decay=0.0001)

    ## start training here! ###
    log.write('** start training here! **\n')
    log.write(' epoch   iter   rate  |  smooth_loss  train_loss  (acc)  |  valid_loss    (acc)   | min\n')
    log.write('---------------------------------------------------------------------------------------\n')

    smooth_loss = 0.0
    train_loss  = np.nan
    train_acc   = np.nan
    test_loss   = np.nan
    test_acc    = np.nan
    time = 0

    for epoch in range(num_epoches):  # loop over the dataset multiple times
        #print ('epoch=%d'%epoch)
        start = timer()

        if 1:
            lr = 0.1 # schduler here ----------------------------
            #if epoch>25: lr=0.01
            #if epoch>35: lr=0.001

            #if epoch>20: lr=0.01
            #if epoch>30: lr=0.001

            if epoch>10: lr=0.05
            if epoch>20: lr=0.01
            if epoch>27: lr=0.005
            if epoch>31: lr=0.001
            if epoch>36: break
            adjust_learning_rate(optimizer, lr)

        rate =  get_learning_rate(optimizer)[0] #check
        #---------------------------------------------------

        sum_smooth_loss = 0.0
        sum = 0
        net.cuda().train()

        num_its = len(train_loader)
        for it, (images, labels, indices) in enumerate(train_loader, 0):

            optimizer.zero_grad()

            logits = net(Variable(images.cuda()))
            loss  = criterion(logits, labels.cuda())
            loss.backward()
            optimizer.step()

            #additional metrics
            sum_smooth_loss += loss.data[0]
            sum += 1

            # print statistics
            if it % it_print == it_print-1:    # print every 2000 mini-batches
                smooth_loss = sum_smooth_loss/sum
                sum_smooth_loss = 0.0
                sum = 0

                train_acc  = f_measure(logits.data, labels.cuda())
                train_loss = loss.data[0]

                print('\r%5.1f   %5d    %0.4f   |  %0.3f  %0.3f  %5.3f | ... ' % \
                        (epoch + it/num_its, it + 1, rate, smooth_loss, train_loss, train_acc),\
                        end='',flush=True)



        end = timer()
        time = (end - start)/60
        if epoch % epoch_test == epoch_test-1  or epoch == num_epoches-1:
            #print('test')

            net.cuda().eval()
            test_loss,test_acc,test_num = evaluate(net, test_loader)
            assert(test_num==test_dataset.num)

            print('\r',end='',flush=True)
            log.write('%5.1f   %5d    %0.4f   |  %0.3f  %0.3f  %5.3f | %0.3f  %5.3f  |  %3.1f min \n' % \
                    (epoch + 1, it + 1, rate, smooth_loss, train_loss, train_acc, test_loss,test_acc, time))

        if epoch % epoch_save == epoch_save-1 or epoch == num_epoches-1:
            torch.save(net,out_dir +'/snap/%03d.torch'%(epoch+1))


    ## check : load model and re-test
    torch.save(net,out_dir +'/snap/final.torch')
    if 1:
        net = torch.load(out_dir +'/snap/final.torch')

        net.cuda().eval()
        test_loss,test_acc,test_num = evaluate(net, test_loader )

        log.write('\n')
        log.write('%s:\n'%(out_dir +'/snap/final.torch'))
        log.write('\ttest_loss=%f, test_acc=%f, test_num=%d\n'%(test_loss,test_acc,test_num))



##to determine best threshold etc ...
def do_others():

    out_dir ='/root/share/project/pytorch/results/kaggle-forest/new-xx05'
    model_file = out_dir +'/snap/final.torch'  #final
    save_dirnames = ['default', 'left-right', 'up-down' ,'rotate',] #'default'  # #'up-down'   #'default' #

    log = Logger()
    log.open(out_dir+'/log.others.txt',mode='a')
    log.write('\n--- [START %s] %s\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    num_classes = 17
    batch_size  = 128
    height,width = SIZE,SIZE

    test_dataset = KgForestDataset('valid-3000',  ##'train-40479', ##'train-39479', ##'valid-1000',  ##
                                    transform=transforms.Compose([
                                        transforms.Lambda(lambda x: toTensor(x)),
                                    ]),
                                    ext=EXT,
                                    height=height,width=width,label_csv='train_label.csv')
    test_loader  = DataLoader(
                        test_dataset,
                        sampler=None,
                        batch_size=batch_size,
                        drop_last=False,
                        num_workers=5,
                        pin_memory=True)

    in_channels = test_dataset.images.shape[3]
    log.write('\t(height,width)    = (%d, %d)\n'%(height,width))
    log.write('\tin_channels       = %d\n'%(in_channels))
    log.write('\tEXT               = %s\n'%(EXT))
    log.write('\ttest_dataset.num = %d\n'%(test_dataset.num))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    log.write('\tmodel_file = %s\n'%model_file)
    log.write('\n')

    net = Net((in_channels,height,width), num_classes)
    net = torch.load(model_file)
    net.cuda().eval()


    ## start experiments here !!!!
    all_thresholds=[]
    all_scores = 0
    for save_dirname in save_dirnames:

        test_dir = out_dir +'/others/'+ save_dirname
        os.makedirs(test_dir, exist_ok=True)
        log.write('\**experiment @ test_dir = %s **\n'%test_dir)
        log.write('\n')

        test_dataset_images = test_dataset.images.copy()

        ## perturb here for test argumnetation  ## ----
        if save_dirname == 'left-right' :
            num=test_dataset.num
            for n in range(num):
                image = test_dataset.images[n]
                test_dataset.images[n] = cv2.flip(image,1)

        if save_dirname == 'up-down' :
            num=test_dataset.num
            for n in range(num):
                image = test_dataset.images[n]
                test_dataset.images[n] = cv2.flip(image,0)

        if save_dirname == 'rotate':
            num=test_dataset.num
            for n in range(num):
                image = test_dataset.images[n]
                test_dataset.images[n] = randomRotate90(image)
        ## ---------------------------------------------

        # do testing here ###
        test_num  = 0
        test_loss = 0
        test_acc  = 0
        predictions=[]
        index=[]
        for iter, (images, labels, indices) in enumerate(test_loader, 0):

            # forward
            logits = net(Variable(images.cuda()))
            loss   = criterion(logits,labels.cuda())
            batch_size = len(images)
            test_acc  += batch_size*f_measure(logits.data, labels.cuda())
            test_loss += batch_size*loss.data[0]
            test_num  += batch_size

            probs = logits_to_probs(logits.data)
            predictions  += probs.cpu().numpy().reshape(-1).tolist()
            index        += indices.numpy().reshape(-1).tolist()

        test_acc  = test_acc/test_num
        test_loss = test_loss/test_num
        log.write('\n')
        log.write('%s:\n'%(model_file))
        log.write('\ttest_loss=%f, test_acc=%f, test_num=%d\n\n'%(test_loss,test_acc,test_num))

        index=np.array(index).reshape(-1)
        predictions=np.array(predictions).reshape(-1, num_classes)
        labels=test_dataset.labels

        np.save(test_dir+'/predictions.npy',predictions)
        np.save(test_dir+'/labels.npy',labels)
        np.savetxt(test_dir+'/predictions.txt',predictions,fmt='%.3f', delimiter=' ', newline='\n')
        np.savetxt(test_dir+'/labels.txt',labels,fmt='%.3f', delimiter=' ', newline='\n')
        np.savetxt(test_dir+'/index.txt',index,fmt='%.0f', delimiter=' ', newline='\n')

        #do threshold adjustment ....
        if 1:
            thresholds, scores = find_f_measure_threshold(predictions, labels)
            i = np.argmax(scores)
            best_threshold, best_score = thresholds[i], scores[i]

            log.write('\tmethod1:\n')
            log.write('\tbest_threshold=%f, best_score=%f\n\n'%(best_threshold, best_score))
            #plot_f_measure_threshold(thresholds, scores)
            #plt.pause(0)

        if 1:
            best_thresholds,  best_scores = find_f_measure_threshold2(predictions, labels)
            log.write('\tmethod2:\n')
            log.write('\tbest_threshold\n')
            log.write (str(best_thresholds)+'\n')
            log.write('\tbest_scores\n')
            log.write (str(best_scores)+'\n')
            log.write('\n')


        all_thresholds.append(best_thresholds)
        all_scores += best_scores[num_classes-1]

        ##revert!
        test_dataset.images= test_dataset_images

    ## final ##
    N=len(save_dirnames)
    all_thresholds = np.sum(np.array(all_thresholds),axis=0)/N
    all_scores = all_scores/N
    log.write('\t**all_thresholds**\n')
    log.write (str(all_thresholds)+'\n')
    log.write('\t**all_scores**\n')
    log.write (str(all_scores)+'\n')



def do_submission():

    out_dir='/root/share/project/pytorch/results/kaggle-forest/new-xx05'
    model_file = out_dir +'/snap/final.torch'  #final
    thresholds = \
        [0.2375,  0.1925,  0.1625,  0.12,    0.2625,  0.245,   0.205,   0.265,   0.2175,  0.21,    0.085,   0.0875,  0.2225,  0.1375,  0.19,    0.14,    0.0475]
    #0.23
    save_dirnames = ['default', 'left-right', 'up-down' ,'rotate',] #'default'  # #'up-down'   #'default' #

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    height,width = SIZE, SIZE
    num_classes  = 17

    test_dataset = KgForestDataset('test-40669',
                                   transform=transforms.Compose([
                                        transforms.Lambda(lambda x: toTensor(x)),
                                   ]),
                                   ext=EXT,
                                   height=height,width=width,label_csv=None)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler=None,
                        batch_size=128,
                        drop_last=False,
                        num_workers=5,
                        pin_memory=True)


    in_channels = test_dataset.images.shape[3]
    log.write('\t(height,width)    = (%d, %d)\n'%(height,width))
    log.write('\tin_channels       = %d\n'%(in_channels))
    log.write('\tEXT               = %s\n'%(EXT))
    log.write('\ttest_dataset.num = %d\n'%(test_dataset.num))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    log.write('\tmodel_file = %s\n'%model_file)
    log.write('\tthresholds= %s\n'%str(thresholds))
    log.write('\n')

    net = Net((in_channels,height,width), num_classes)
    net = torch.load(model_file)
    net.cuda().eval()


    # do prediction here !!!!
    for save_dirname in save_dirnames:
        sub_dir=out_dir +'/submission/'+ save_dirname
        os.makedirs(sub_dir, exist_ok=True)
        log.write('\**submission @ sub_dir = %s **\n'%sub_dir)
        log.write('\n')

        test_dataset_images = test_dataset.images.copy()

        ## perturb here for test argumnetation  ## ----
        if save_dirname == 'left-right' :
            num=test_dataset.num
            for n in range(num):
                image = test_dataset.images[n]
                test_dataset.images[n] = cv2.flip(image,1)

        if save_dirname == 'up-down' :
            num=test_dataset.num
            for n in range(num):
                image = test_dataset.images[n]
                test_dataset.images[n] = cv2.flip(image,0)

        if save_dirname == 'rotate':
            num=test_dataset.num
            for n in range(num):
                image = test_dataset.images[n]
                test_dataset.images[n] = randomRotate90(image)
        ## ---------------------------------------------


        n = 0
        predictions=[]
        index=[]
        for iter, (images, indices) in enumerate(test_loader, 0):
            batch_size = len(images)
            n += batch_size
            print('\riter=%d:  %d/%d'%(iter,n,test_dataset.num),end='',flush=True)

            # forward
            logits = net(Variable(images.cuda()))
            probs = logits_to_probs(logits.data)

            predictions  += probs.cpu().numpy().reshape(-1).tolist()
            index += indices.numpy().reshape(-1).tolist()
        print('')

        predictions = np.array(predictions).reshape(-1,num_classes)
        index = np.array(index).reshape(-1)
        num_test = len(predictions)
        assert(num_test==test_dataset.num)

        # write to csv
        classes = test_dataset.classes
        names   = test_dataset.names
        images  = test_dataset.images
        ext  = test_dataset.ext

        with open(sub_dir+'/results.csv','w') as f:
            f.write('image_name,tags\n')
            for n in range(num_test):
                name = names[index[n]]
                shortname = name.split('/')[-1].replace('.'+ext,'')

                prediction = predictions[n]
                s = get_multi_classes(prediction, classes, threshold=thresholds)
                f.write('%s,%s\n'%(shortname,s))


                # save to images
                # print('%32s : [%s] %s  %s, %s'% \
                #       (name, weather_label, other_label.T, weather_classes[weather_label], s))
                #
                #
                # image = images[index[n]]
                # h,w = image.shape[0:2]
                # if 256!=h or 256!=w:
                #     image=cv2.resize(image,(256,256))
                #
                # draw_shadow_text  (image, shortname, (5,15),  0.5, (255,255,255), 1)
                # draw_multi_classes(image, weather_classes, other_classes, weather_label, other_label)
                # im_show('image',image)
                # cv2.waitKey(0)


        np.save(sub_dir+'/predictions.npy',predictions)
        np.savetxt(sub_dir+'/predictions.txt',predictions,fmt='%.3f', delimiter=' ', newline='\n')
        np.savetxt(sub_dir+'/index.txt',index,fmt='%.0f', delimiter=' ', newline='\n')

        ##revert!
        test_dataset.images= test_dataset_images


def do_averaging():

    thresholds = \
        [0.2375,  0.1925,  0.1625,  0.12,    0.2625,  0.245,   0.205,   0.265,   0.2175,  0.21,    0.085,   0.0875,  0.2225,  0.1375,  0.19,    0.14,    0.0475]

    ave_dir='/root/share/project/pytorch/results/kaggle-forest/new-xx05/submission/average'
    os.makedirs(ave_dir, exist_ok=True)
    log = Logger()
    log.open(ave_dir+'/log.average.txt',mode='a')


    predict_files = [
        '/root/share/project/pytorch/results/kaggle-forest/new-xx05/submission/default/predictions.npy',
        '/root/share/project/pytorch/results/kaggle-forest/new-xx05/submission/left-right/predictions.npy',
        '/root/share/project/pytorch/results/kaggle-forest/new-xx05/submission/up-down/predictions.npy',
        '/root/share/project/pytorch/results/kaggle-forest/new-xx05/submission/rotate/predictions.npy',
    ]
    log.write('%s\n'%(predict_files))
    log.write('threshold=%s\n'%str(thresholds))

    predictions = np.load(predict_files[0])
    num=len(predict_files)
    for n in range(1,num):
        predictions += np.load(predict_files[n])
    predictions = predictions/num

    if 1:
        with open('/root/share/data/kaggle-forest/classes') as f:
            classes = f.readlines()
        classes = [x.strip() for x in classes]

        num_test=len(predictions)
        with open(ave_dir+'/results.csv','w') as f:
            f.write('image_name,tags\n')
            for n in range(num_test):

                shortname = 'test_%d'%n
                prediction = predictions[n]
                s = get_multi_classes(prediction, classes, threshold=thresholds)
                f.write('%s,%s\n'%(shortname,s))





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #do_training()
    #do_others()


    #do_submission()
    do_averaging()

    print('sucess')
