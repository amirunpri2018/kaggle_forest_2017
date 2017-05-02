#  multi label loss:
#    https://discuss.pytorch.org/t/calculating-accuracy-for-a-multi-label-classification-problem/2303
#    https://gist.github.com/bartolsthoorn/36c813a4becec1b260392f5353c8b7cc


from net.common import *
from net.dataset.tool import *
from net.utility.tool import *

from net.dataset.kgforest import *
from net.dataset.standard import *
from net.model.vggnet import VggNet as Net


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

## <todo>
#   this is  using the 'micro' flag which is wrong!!!!
#    f0 = fbeta_score(labels, probs, beta=2, average='micro')  #micro  #samples
def f_measure(outputs, labels, threshold=0, beta=2 ):

    batch_size, num_classes = labels.size()[0],labels.size()[1]
    l = labels.byte()
    o = outputs>threshold

    num_pos     = o.sum() + 1e-5
    num_pos_hat = l.sum()

    tp = torch.mul(l==o , l).sum()
    precise = tp/num_pos
    recall  = tp/num_pos_hat
    acc = (1+beta*beta)*precise*recall/(beta*beta*precise + recall + 1e-5)

    return acc


def evaluate(net, test_loader, criterion, accuracy ):

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (images, labels, indices) in enumerate(test_loader, 0):

        # forward + backward + optimize
        outputs = net(Variable(images.cuda()))
        loss    = criterion(outputs, Variable(labels.cuda()))

        batch_size = len(images)

        test_acc  += batch_size*accuracy(outputs.data, labels.cuda())
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num

    return test_loss,test_acc,test_num








def do_training():

    out_dir='/root/share/project/pytorch/results/kaggle-forest/xx13'
    initial_model = None #None '/root/share/project/pytorch/results/kaggle-forest/xx11/snap/final.torch'

    os.makedirs(out_dir,exist_ok=True)
    os.makedirs(out_dir +'/snap', exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n')
    log.write('--- [START %s] %s\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')
    log.write('** some experiment setting **\n')
    log.write('\tSEED    = %u\n' % SEED)
    log.write('\tfile    = %s\n' % __file__)
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')
    #initial_model = None


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    num_classes = 17
    batch_size  = 128 #+32
    sgd_size    = batch_size
    height= 64  #48 #64  #56
    width = height#64-8


    default_transform = [ transforms.ToTensor(), ]
    train_transform = [
                    transforms.RandomHorizontalFlip(), # ... <todo>... e.g. rotation, scale ???
                    #transforms.RandomCrop(224),
                ]
    test_transform = [
                    #transforms.CenterCrop(224),
                ]


    train_dataset = KgForestDataset('train-39479', #train-39479
                                    transform=transforms.Compose(train_transform+default_transform),
                                    height=height,width=width)
    train_loader  = DataLoader(
                        train_dataset,
                        sampler=RandomSampler(train_dataset),
                        batch_size=batch_size,
                        drop_last=True,
                        num_workers=4,
                        pin_memory=True)

    test_dataset = KgForestDataset('valid-1000',
                                   transform=transforms.Compose(test_transform+default_transform),
                                   height=height,width=width)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler=None,
                        batch_size=batch_size,
                        drop_last=False,
                        num_workers=4,
                        pin_memory=True)

    log.write('\t(height,width) = (%d, %d)\n'%(height,width))
    log.write('\ttrain_dataset.num = %d\n'%(train_dataset.num))
    log.write('\ttest_dataset.num  = %d\n'%(test_dataset.num))
    log.write('\tdefault_transform = %s\n'%str(default_transform))  #<todo> log transform
    log.write('\ttrain_transform   = %s\n'%str(train_transform))
    log.write('\ttest_transform    = %s\n'%str(test_transform))
    log.write('\n')

    if 0:
        ## check data
        check_kgforest_dataset(train_dataset, train_loader)
        exit(0)

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(in_channels=3, num_classes=num_classes)
    if initial_model is not None:
        net = torch.load(initial_model)

    net.cuda()
    log.write('\n%s\n'%(str(net)))
    log.write('\n')


    ## optimiser ----------------------------------
    num_epoches=45
    it_print  =10
    epoch_test=1
    epoch_save=1

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.SGD(net.parameters(), lr=0, momentum=0.9, weight_decay=0.0005)

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

        lr = 0.1 # schduler here
        if epoch>25: lr=0.01
        if epoch>35: lr=0.001

        adjust_learning_rate(optimizer, lr)
        rate =  get_learning_rate(optimizer)[0] #check

        ##https://discuss.pytorch.org/t/how-to-implement-accumulated-gradient-in-pytorch-i-e-iter-size-in-caffe-prototxt/2522
        ##https://discuss.pytorch.org/t/pytorch-gradients/884
        sum_smooth_loss = 0.0
        sum = 0
        net.cuda().train()

        #
        # train_iterator  = iter(train_loader)
        # it_size = sgd_size//batch_size
        # num_its = train_dataset.num//batch_size//it_size
        #
        # for it in range(num_its):
        #     optimizer.zero_grad()
        #
        #     loss = 0
        #     for it1 in range (it_size):
        #         #print(it1)
        #         (images, labels, indices) = train_iterator.next()
        #         outputs = net(Variable(images.cuda()))
        #         loss    += criterion(outputs, Variable(labels.cuda()))
        #         loss.backward(retain_variables=(it1!=it_size-1))
        #
        #     optimizer.step()
        #     loss = loss/it_size

        num_its = len(train_loader)
        for it, (images, labels, indices) in enumerate(train_loader, 0):

            optimizer.zero_grad()
            outputs = net(Variable(images.cuda()))
            loss    = criterion(outputs, Variable(labels.cuda()))
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

                train_acc  = f_measure(outputs.data, labels.cuda())
                train_loss = loss.data[0]

                print('\r%5.1f   %5d    %0.4f   |  %0.3f  %0.3f  %5.3f | ... ' % \
                        (epoch + it/num_its, it + 1, rate, smooth_loss, train_loss, train_acc),\
                        end='',flush=True)



        end = timer()
        time = (end - start)/60
        if epoch % epoch_test == epoch_test-1  or epoch == num_epoches-1:
            #print('test')

            net.cuda().eval()
            test_loss,test_acc,test_num = evaluate(net, test_loader, criterion, f_measure )
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
        test_loss,test_acc,test_num = evaluate(net, test_loader, criterion, f_measure )

        log.write('\n')
        log.write('%s:\n'%(out_dir +'/snap/final.torch'))
        log.write('\ttest_loss=%f, test_acc=%f, test_num=%d\n'%(test_loss,test_acc,test_num))


#  to do:  weather_labels = ['clear', 'haze', 'partly_cloudy','cloudy'  ]
def modify_for_weather(score):
    i=np.argmax(score[0:4])
    score[0:4]=-10
    score[i]=10


def do_predicting():

    out_dir='/root/share/project/pytorch/results/kaggle-forest/xx13'
    os.makedirs(out_dir +'/submission', exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.predict.txt',mode='a')
    log.write('\n')
    log.write('--- [START %s] %s\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    height,width=64,64
    num_classes=17

    default_transform = [transforms.ToTensor(), ]
    test_transform = [
                    #transforms.CenterCrop(224),
                ]


    test_dataset = CropDataset('test-40669',
                               data_dir = '/root/share/data/kaggle-forest',
                               transform=transforms.Compose(test_transform+default_transform),
                               height=height,width=width)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler=None,
                        batch_size=64,
                        drop_last=False,
                        num_workers=4,
                        pin_memory=True)

    log.write('\t(height,width) = (%d, %d)\n'%(height,width))
    log.write('\ttest_dataset.num = %d\n'%(test_dataset.num))
    log.write('\tdefault_transform = %s\n'%str(default_transform))
    log.write('\ttest_transform    = %s\n'%str(test_transform))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(in_channels=3, num_classes=num_classes)
    net = torch.load(out_dir +'/snap/final.torch')
    #net = torch.load(out_dir +'/snap/044.torch')
    net.cuda().eval()


    # do prediction here ###
    n = 0
    predictions=[]
    index=[]
    for iter, (images, indices) in enumerate(test_loader, 0):
        batch_size = len(images)
        n += batch_size
        print('\riter=%d:  %d/%d'%(iter,n,test_dataset.num),end='',flush=True)

        # forward + backward + optimize
        outputs = net(Variable(images.cuda()))

        predictions  += outputs.data.cpu().numpy().reshape(-1).tolist()
        index += indices.numpy().reshape(-1).tolist()
    print('')

    predictions = np.array(predictions).reshape(-1,num_classes)
    num_test = len(predictions)
    assert(num_test==test_dataset.num)

    # write to csv
    classes = test_dataset.classes
    names = test_dataset.names

    with open(out_dir+'/submission/results.csv','w') as f:
        f.write('image_name,tags\n')
        for n in range(num_test):
            name = names[index[n]]
            shortname = name.split('/')[-1].replace('.jpg','')

            prediction = predictions[n]
            modify_for_weather(prediction)

            s = get_multi_classes(prediction, classes, threshold=-1.25)
            f.write('%s,%s\n'%(shortname,s))





def do_testing():

    out_dir='/root/share/project/pytorch/results/kaggle-forest/xx6'
    os.makedirs(out_dir +'/test', exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.test.txt',mode='a')
    log.write('\n')
    log.write('--- [START %s] %s\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    num_classes=17
    default_transform = [transforms.ToTensor(), ]
    test_transform = [
                    #transforms.CenterCrop(224),
                ]

    test_dataset = KgForestDataset('valid-1000',
                               transform=transforms.Compose(test_transform+default_transform),
                               height=64,width=64)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler=None,
                        batch_size=64,
                        drop_last=False,
                        num_workers=4,
                        pin_memory=True)

    log.write('\ttest_dataset.num = %d\n'%(test_dataset.num))
    log.write('\tdefault_transform = %s\n'%str(default_transform))
    log.write('\ttest_transform    = %s\n'%str(test_transform))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(in_channels=3, num_classes=num_classes)
    #net = torch.load(out_dir +'/snap/final.torch')
    net = torch.load(out_dir +'/snap/021.torch')
    net.cuda().eval()


    # do testing here ###
    criterion = nn.MultiLabelSoftMarginLoss()
    test_loss,test_acc,test_num = evaluate(net, test_loader, criterion, f_measure )

    log.write('\n')
    log.write('%s:\n'%(out_dir +'/snap/final.torch'))
    log.write('\ttest_loss=%f, test_acc=%f, test_num=%d\n'%(test_loss,test_acc,test_num))


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

     #do_training()


    do_predicting()
    #do_testing()

    print('sucess')
