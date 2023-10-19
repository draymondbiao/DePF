import torch
from torch import nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np

class reflect_conv(nn.Module):#共享权重
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),#边界填充，与零填充相比， 填充内容来自输入
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


def gradient(input):
    """
    求图像梯度, sobel算子
    :param input:
    :return:
    """

    filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter1.weight.data = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]).reshape(1, 1, 3, 3).cuda()
    # ]).reshape(1, 1, 3, 3).to(device)

    filter2.weight.data = torch.tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ]).reshape(1, 1, 3, 3).cuda()
    # ]).reshape(1, 1, 3, 3).to(device)

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)
    return image_gradient


def gradient_lp(input):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(1, 1, 3, bias=False,padding=0)
    # 定义算子参数 [0.,1.,0.],[1.,-4.,1.],[0.,1.,0.] Laplacian 四邻域 八邻域
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # 将算子转换为适配卷积操作的卷积核
    kernel = kernel.reshape((1, 1, 3, 3))
    # 给卷积操作的卷积核赋值
    conv_op.weight.data = (torch.from_numpy(kernel)).cuda().type(torch.float32)
    # 对图像进行卷积操作
    edge_detect = conv_op(input)
    return edge_detect

def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)
#

def gradient_UNI(x):
    x = x.cuda()
    kernel = [[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    weight = nn.Parameter(data=kernel,requires_grad=False)

    weight = weight.cuda()
    gradMap = F.conv2d(x,weight=weight,stride=1,padding=1)
    #showTensor(gradMap);
    return gradMap

def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式

    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr

def RGB2YCrCbto(rgb_image):
    """
    将RGB格式转换为YCrCb格式

    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[1:2]
    G = rgb_image[2:3]
    B = rgb_image[3:4]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr

def YCrCb2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式

    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out
import torch.nn.functional as F
def con2(img):
    b = img.size()[0]
    window=torch.ones(size=(1,1,3,3)).to(img.device)
    # window = torch.ones(size=(1, 1, 3, 3))
    from torch.autograd import Variable
    window = Variable(window.expand(1, 1, 3, 3).contiguous())
    con1=F.conv2d(img,window,padding=1)
    return con1


# attention fusion strategy, average based on weight maps
def attention_fusion_weight(tensor1, tensor2):
    # avg, max, nuclear
    f_spatial = spatial_fusion(tensor1, tensor2)
    tensor_f = f_spatial
    return tensor_f

EPSILON = 1e-5

def spatial_fusion(tensor1, tensor2, spatial_type='sum'):
    shape = tensor1.size()
    # calculate spatial attention
    tensor1 = tensor1.to(torch.float64)
    tensor2 = tensor2.to(torch.float64)
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)

    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    # spatial_w1 = spatial1 / (spatial1 + spatial2 + EPSILON)
    # spatial_w2 = spatial2 / (spatial1 + spatial2 + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    # fusion = torch.sum(spatial_w1, dim=0)
    # fusion = torch.sum(fusion, dim=0)
    # fusion = fusion / spatial_w1.size()[1]
    # from torchvision import transforms
    # rgb_fused_image = transforms.ToPILImage()(fusion)
    # rgb_fused_image.save('/data/Disk_A/yongbiao/Fusion/Wave_Transfer/1.png')
    #
    # fusion = torch.sum(spatial_w2, dim=0)
    # fusion = torch.sum(fusion, dim=0)
    # fusion = fusion / spatial_w1.size()[1]
    # from torchvision import transforms
    # rgb_fused_image = transforms.ToPILImage()(fusion)
    # rgb_fused_image.save('/data/Disk_A/yongbiao/Fusion/Wave_Transfer/2.png')


    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f.to(torch.float32)


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial





def channel_f(f1,f2,is_test=False,save_mat=False):
    if is_test:
        fp1 = (((f1.mean(2)).mean(2)).unsqueeze(2)).unsqueeze(3)
        fp2 = (((f2.mean(2)).mean(2)).unsqueeze(2)).unsqueeze(3)
    else:
        fp1 = F.avg_pool2d(f1, f1.size(2))
        fp2 = F.avg_pool2d(f2, f2.size(2))
    mask1 = fp1 / (fp1 + fp2)
    mask2 = 1 - mask1
    if save_mat:
        import scipy.io as io
        mask = mask1.cpu().detach().numpy()
        io.savemat("./outputs/fea/mask.mat", {'mask': mask})
    return f1 * mask1 + f2 * mask2
