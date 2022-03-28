import argparse
from email.policy import default

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
# parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
#                     choices=model_names,
#                     help='model architecture: ' + ' | '.join(model_names) +
#                     ' (default: resnet32)')
parser.add_argument("-m", dest="model_name", type=str, help="choosing model name")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed')
parser.add_argument('--lp', type=str, default="test.log",
                    help='Log file name')
parser.add_argument("--lm", dest="lr_milestones", nargs='+', type=int, help="Assign the mile stones where learning rate decays", default=[100, 150]) 
parser.add_argument("--nc", dest="num_classes", type=int, default=100, help="number of classification objects")                  
parser.add_argument("--en", dest="experiment_name", type=str, default="Experiment_test")
parser.add_argument("--nt", dest="norm_type", type=str, default="NoNorm")
parser.add_argument("--if-save", dest="if_save", type=bool, default=True)
parser.add_argument("--blocks-per-stage", dest="blocks_per_stage", type=int, default=3)
parser.add_argument("--kernel-size", dest="kernel_size", type=int, default=3)
parser.add_argument("--stages", dest="stages", type=int, default=3)