## some processing tools

from net.common import *
from net.dataset.tool import *
from net.utility.tool import *

from net.dataset.kgforest import *
from net.dataset.standard import *
from net.model.vggnet import VggNet as Net

from sklearn.metrics import fbeta_score



## https://www.kaggle.com/anokas/planet-understanding-the-amazon-from-space/fixed-f2-score-in-python/comments
## https://www.kaggle.com/c/planet-understanding-the-amazon-from-space#evaluation


def f_measure(probs, labels, threshold=0.5, beta=2 ):

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


#https://www.kaggle.com/paulorzp/planet-understanding-the-amazon-from-space/find-best-f2-score-threshold/code
def find_f_measure_threshold(probs, labels, thresholds=None):

    best_threshold =  0
    best_score     = -1

    if thresholds is None:
        thresholds = np.arange(0,1,0.005)
        ##thresholds = np.unique(probs)

    for t in thresholds:
        score = f_measure(probs, labels, threshold=t)
        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold,best_score




def run_test_fscore():

    y = [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
         [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]

    p = [[0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
         [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
         [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]

    labels = np.array(y, dtype=np.float32)
    probs  = np.array(p, dtype=np.float32)
    f0 = fbeta_score(labels, probs, beta=2, average='samples')  #micro  #samples
    f1 = f_measure(probs, labels)


    print ('f0=%f'%f0)
    print ('f1=%f'%f1)
    pass

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_test_fscore()

    print('sucess')
