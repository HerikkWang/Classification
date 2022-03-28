import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def position_coding(pos1, pos2):
    char_dict = {}
    for i in range(len(pos1)):
        char_dict[pos1[i]] = i
    
    code = []
    for i in range(len(pos2)):
        code.append(char_dict[pos2[i]])
    
    return tuple(code)

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)