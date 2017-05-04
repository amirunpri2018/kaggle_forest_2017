from net.common import *
from net.dataset.tool import *
from net.utility.tool import *

from net.dataset.cifar10 import *
from net.model.dummynet import DummyNet as Net


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr


def accuracy(outputs, labels, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = len(labels)

    _, indices = outputs.topk(maxk, 1, True, True)
    indices = indices.t()
    corrects = indices.eq(labels.view(1, -1).expand_as(indices))

    results = []
    for k in topk:
        correct_k = corrects[:k].view(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / batch_size))

    return results


def evaluate(net, test_loader, criterion, accuracy ):

   test_loss = 0
   test_acc  = 0
   test_num  = 0
   for iter, (images, labels) in enumerate(test_loader, 0):

        # forward + backward + optimize
        outputs = net(Variable(images.cuda()))
        loss    = criterion(outputs, Variable(labels.cuda()))

        batch_size = len(images)
        test_acc  += batch_size*accuracy(outputs.data, labels.cuda(), topk=(1,))[0].cpu()[0]
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

   test_acc  = test_acc/test_num
   test_loss = test_loss/test_num

   return test_loss,test_acc,test_num



def do_training():

    out_dir='/root/share/project/pytorch/results/xx0'
    os.makedirs(out_dir,exist_ok=True)
    os.makedirs(out_dir +'/snap', exist_ok=True)

    log = Logger(out_dir+'/log.txt',mode='a')
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
    default_transform = [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range in [-1,1]
                ]
    augment_transform = [
                    # ... <todo>...
                ]

    train_dataset = CifarDataset('train', transform=transforms.Compose(default_transform+augment_transform))
    train_loader  = DataLoader(
                        train_dataset,
                        sampler=RandomSampler(train_dataset),
                        batch_size=32,
                        drop_last=True,
                        num_workers=4,
                        pin_memory=True)

    test_dataset = CifarDataset('test', transform=transforms.Compose(default_transform))
    test_loader  = DataLoader(
                        test_dataset,
                        sampler=None,
                        batch_size=32,
                        drop_last=False,
                        num_workers=4,
                        pin_memory=True)

    log.write('\ttrain_dataset.num = %d\n'%(train_dataset.num))
    log.write('\ttest_dataset.num = %d\n'%(test_dataset.num))
    log.write('\tdefault_transform = %s\n'%str(default_transform))  #<todo> log transform
    log.write('\taugment_transform = %s\n'%str(augment_transform))

    log.write('\n')

    if 0:
        ## check data
        pass

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net()
    net.cuda()

    log.write('\n%s\n'%(str(net)))
    log.write('\n')


    ## optimiser ----------------------------------
    num_epoches=50
    iter_print=100
    epoch_test=2
    epoch_save=2

    criterion = nn.CrossEntropyLoss()
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

        lr = 0.001 # schduler here
        adjust_learning_rate(optimizer, lr)
        rate =  get_learning_rate(optimizer) #check

        sum_smooth_loss = 0.0
        sum = 0
        for iter, (images, labels) in enumerate(train_loader, 0):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(Variable(images.cuda()))
            loss    = criterion(outputs, Variable(labels.cuda()))
            loss.backward()
            optimizer.step()


            #additional metrics
            sum_smooth_loss += loss.data[0]
            sum += 1

            # print statistics
            if iter % iter_print == iter_print-1:    # print every 2000 mini-batches
                smooth_loss = sum_smooth_loss/sum
                sum_smooth_loss = 0.0
                sum = 0

                train_acc  = accuracy(outputs.data, labels.cuda(), topk=(1,))[0].cpu()[0]
                train_loss = loss.data[0]

                log.write('\r')
                log.write('%3d   %5d    %0.4f   |  %0.3f  %0.3f  %4.1f | ............ ' % \
                        (epoch + 1, iter + 1, rate[0], smooth_loss, train_loss, train_acc),\
                        is_file=0)



        end = timer()
        time = (end - start)/60
        if epoch % epoch_test == epoch_test-1  or epoch == num_epoches-1:
            test_loss,test_acc,test_num = evaluate(net, test_loader, criterion, accuracy )
            assert(test_num==test_dataset.num)


            log.write('\r')
            log.write('%3d   %5d    %0.4f   |  %0.3f  %0.3f  %4.1f | %0.3f  %4.1f  |  %3.1f min' % \
                    (epoch + 1, iter + 1, rate[0], smooth_loss, train_loss, train_acc, test_loss,test_acc, time),\
                    is_file=1)
            log.write('\n')


        if epoch % epoch_save == epoch_save-1 or epoch == num_epoches-1:
            torch.save(net,out_dir +'/snap/%03d.torch'%epoch)


    ## check : load model and re-test
    torch.save(net,out_dir +'/snap/final.torch')
    if 1:
        net = torch.load(out_dir +'/snap/final.torch')
        net.cuda()
        test_loss,test_acc,test_num = evaluate(net, test_loader, criterion, accuracy )

        log.write('\n')
        log.write('%s:\n'%(out_dir +'/snap/final.torch'))
        log.write('\ttest_loss=%f, test_acc=%f, test_num=%d\n'%(test_loss,test_acc,test_num))


    log.write('\n')
    log.write('sucess!\n')


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    do_training()
