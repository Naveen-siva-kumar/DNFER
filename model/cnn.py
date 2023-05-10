'''
Aum Sri Sai Ram

Naveen
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from model.resnet import *
import pickle
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
def resModel(args): #resnet18
    
    model = torch.nn.DataParallel(resnet18(end2end= False,  pretrained= False)).to(device)
    
    if args is None  or args.pretrained:
       
       checkpoint = torch.load(args.pretrained)
       pretrained_state_dict = checkpoint['state_dict']
       model_state_dict = model.state_dict()
        
       for key in pretrained_state_dict:
           if  ((key == 'module.fc.weight') | (key=='module.fc.bias') | (key=='module.feature.weight') | (key=='module.feature.bias') ) :
               print(key) 
               pass
           else:
               #print(key)
               model_state_dict[key] = pretrained_state_dict[key]

       model.load_state_dict(model_state_dict, strict = False)
       print('Model loaded from Msceleb pretrained')
    else:
       print('No pretrained resent18 model built.')
    return model   

def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    #print(own_state.keys())
    for name, param in weights.items():
        new_name = 'module.' + name 
        if new_name in own_state and (  name.find('fc')<=-1): #don't load layer4 +fc name.find('layer4')<=-1 and
            
            try:
                #print('copying', new_name)
                own_state[new_name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
    print('\nModel loaded with ', fname)      

def resModel_50(args): #resnet50
    
    model = torch.nn.DataParallel(resnet50(end2end= False,  pretrained= True)).to(device)

    if args.pretrained:
       
       load_state_dict(model,'pretrained/vgg_msceleb_resnet50_ft_weight.pkl')
       print('Resnet50 Model loaded from Msceleb pretrained')
    else:
       print('No pretrained resnet50 model built.')
    return model   
