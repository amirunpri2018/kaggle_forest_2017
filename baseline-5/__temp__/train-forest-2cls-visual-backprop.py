## https://github.com/colesbury/examples/blob/8a19c609a43dc7a74d1d4c71efcda102eed59365/imagenet/main.py


## VisualBackProp: visualizing CNNs for autonomous driving
#  Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car
## https://arxiv.org/pdf/1611.05418.pdf
## https://arxiv.org/pdf/1704.07911.pdf
##    https://github.com/mbojarski/VisualBackProp
##

from net.common import *
from net.dataset.tool import *
from net.utility.tool import *

from net.dataset.kgforest2cls import *
from sklearn.metrics import fbeta_score


from net.model.simplenet_2cls import SimpleNet_2cls_2 as Net



##https://github.com/torch/torch7/wiki/Torch-for-Numpy-users

## global setting -------------
CLASS_NAME='road'
CLASS_NO  =9

EXT  = 'jpg'  #'jpg'  #'tif'  #'jpg'  #
SIZE = 96  ##112  #64
IN_CHANNELS = 3




##-----------------------------
## extract feature maps using hook
## http://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#forward-and-backward-function-hooks
## https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119
def hook_net(net):
    maps = []
    layers=[]
    hooks =[]

    def fun(module, inputs, outputs):
        print (module)
        print ('\t'+str(inputs[0].data.size()))
        print ('\t'+str(outputs[0].data.size()))
        print ('')

        ## copy data out ##
        map = outputs[0].data.cpu()
        maps.append(map)


    keys0 =net._modules.keys()
    #print(keys0)
    for k0 in keys0:
        m = net._modules.get(k0)
        if type(m)==torch.nn.modules.container.Sequential:
            keys1 = m._modules.keys()
            #print(keys1)

            for k1 in keys1:
                mm = m._modules.get(k1)
                type_name = str(type(mm)).replace("<'",'').replace("'>",'').split('.')[-1]
                name = k0 + '-' + k1 + '-' + type_name

                hook = mm.register_forward_hook(fun)
                layers.append((name,mm))
                hooks.append(hook)
                #print(name)

    return layers, maps, hooks



def unhook_net(net, hooks):
    for hook in hooks:
        hook.remove()



def save_feature_maps(save_dir, img, layers, maps):

    os.makedirs(save_dir,exist_ok=True)
    cv2.imwrite(save_dir + '/input.png', img)

    num_layers = len(layers)
    for n in range(num_layers):
        layer = layers[n]
        os.makedirs(save_dir+'/'+layer[0],exist_ok=True)

        map = maps[n].numpy()
        if map.ndim==1:
            l = len(map)
            if l==1:
                with open(save_dir+'/'+ layer[0] + '/%f'%map[0], 'w') as f:
                    pass

            else:
                w = int(l**0.5)
                h = int(np.ceil(l/w))

                f = map
                fmax=np.max(f)
                fmin=np.min(f)
                f = ((f-fmin)/(fmax-fmin+1e-12)*255).astype(np.uint8)

                f1 = np.zeros(h*w, np.uint8)
                f1[:l] = f
                f1 = f1.reshape(h,w)
                cv2.imwrite(save_dir+'/'+ layer[0] + '/out.png', f1)


        else: # assume  map.ndim==3
            #save txt for debug
            np.savetxt(save_dir+'/'+ layer[0] + 'out', map[0], fmt='%0.6f', delimiter='\t', newline='\n')

            num_channels =len(map)
            for c in range(num_channels):
                f = map[c]
                fmax=np.max(f)
                fmin=np.min(f)
                f = ((f-fmin)/(fmax-fmin+1e-12)*255).astype(np.uint8)
                cv2.imwrite(save_dir+'/'+ layer[0] + '/out%03d.png'%c, f)




def make_contribution(image, layers, maps):

    ## visual back propagation
    mask=None
    ups=[]
    aves=[]
    num_layers = len(layers)
    for n in range(num_layers-1,0,-1):
        layer=layers[n][1]
        if  type(layer) in [torch.nn.modules.conv.Conv2d]:
            #assert(type(layers[n + 2][1]) in [torch.nn.modules.activation.PReLU])
            assert(type(layers[n + 1][1]) in [torch.nn.BatchNorm2d])
            assert(type(layers[n + 2][1]) in [torch.nn.modules.activation.ReLU])


            input = maps[n-1]
            conv  = maps[n  ]
            bn    = maps[n+1]
            relu  = maps[n+2]
            ave   = relu.mean(dim=0) #np.expand_dims (relu.mean(axis=0),0)
            aves.append(ave)
            if mask is not None:
                mask = mask*ave
            else:
                mask = ave

            oC, oH, oW = list(conv.size())
            iC, iH, iW = list(input.size())

            #  assume only stride 2 ----------------------------------
            #<todo> generalise later ????
            padding=0
            output_padding=0
            stride=1
            if oH!=iH:
                stride = 2  ## iH//oH
                padding=padding=1
                output_padding =1
            #---------------------------------------------------------

            kH, kW = layer.kernel_size
            #upsampling : see http://pytorch.org/docs/nn.html#convtranspose2d
            weight = nn.Parameter(torch.ones( 1, 1, kH, kW ))
            up   = F.conv_transpose2d(Variable(mask.unsqueeze(0)),weight, stride=stride, padding=1, output_padding=1)
                     ##conv_transpose2d (input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1)


            mask = up.squeeze(0).data
            ups.append(mask)
            pass

    contribution = mask
    return contribution

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #output
    save_dir = '/root/share/project/pytorch/results/out/xx'
    os.makedirs(save_dir,exist_ok=True)




    #load a model ---
    model_file = '/root/share/project/pytorch/results/kaggle-forest/2cls-baseline-01/snap/final.torch'
    net = torch.load(model_file)
    net.cuda().eval()




    #load image ---
    split    = 'valid-road-8095'
    ext = EXT  #'jpg'
    height, width = SIZE,SIZE

    data_dir = '/root/share/data/kaggle-forest/classification'
    list_file = data_dir +'/split/'+ split
    with open(list_file) as f:
        lines = f.readlines()
    lines = [x.strip().replace('<ext>',ext) for x in lines]
    num   = len(lines)

    for n in range(num):
        s = lines[n].split(' ')
        name=s[0]
        label=int(s[1])
        img_file  = data_dir + '/image/' + name

        #img_file = '/root/share/data/kaggle-forest/segmentation/image/road/set01/train_26560.jpg'
        #img_file = '/root/share/data/kaggle-forest/classification/image/train-jpg/train_26560.jpg'  ##train_1003

        img = read_and_resize_image(img_file, width, height)

        ## run an image #
        #http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
        image = toTensor(img)
        image = image.unsqueeze(0)  #fake batch dimension.

        layers, maps, hooks = hook_net(net)
        prob  = net.forward(Variable(image.cuda()))
        #print(prob.data.cpu().numpy())
        contribution = make_contribution(image, layers, maps)
        prob = prob.data.cpu().numpy()[0]

        #save contrubutions
        if 1:
            results = np.zeros((height,3*width,3),np.uint8)
            c = np.squeeze( contribution.numpy() )
            cmax = np.max(c)
            cmin = np.min(c)
            c = (prob*(c-cmin)/(cmax-cmin)*255).astype(np.uint8)

            img_c = img.copy()
            draw_mask(img_c, c, color=(255,0,255), α=0.8,  β=1, threshold=None )

            results[:, :width, :]=np.dstack((c,c,c))
            results[:, width:2*width, :]=img_c
            results[:, 2*width:, :]=img
            results=cv2.resize(results,(3*150,150))


            shortname = name.split('/')[-1].replace('.'+ext,'')
            draw_shadow_text(results,'%s (road=%d)'%(shortname,label), (5,15),0.5,(255,255,255),1)
            draw_shadow_text(results,'p=%0.3f'%(prob), (5,30),0.5,(255,255,0),1)

            # im_show('c',c,2)
            # im_show('img',img,2)
            # im_show('img_c',img_c,2)
            im_show('results',results,2)
            cv2.imwrite(save_dir +'/'+shortname+'.png',results)
            cv2.waitKey(1   )

        #save feature maps
        if n==2:
            save_feature_maps(save_dir+'/one', img, layers, maps)

    print('sucess')






