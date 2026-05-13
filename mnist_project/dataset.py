import torch
from torch.utils.data import Dataset , DataLoader
import torchvision.transforms as transforms
from torchvision import datasets


class MNISTDataset(Dataset):
    # root: 数据集存储路径 ./ 表示当前文件所在的文件夹，也就是minst_project
    # train: 是否加载训练集，默认为True  记录是否加载训练集，之后可视化里面会用到
    # transform: 数据预处理方法，默认为None  记录数据预处理方法（把数据转换成模型需要的格式）

    def __init__(self, root='./data', train=True, transform=None):
        #规范性代码 ， 就记住 继承父类 就要调用父类的构造函数
        super().__init__()
        
        #这里把数据加载到内存中，之后的__len__ 和 __getitem__方法就可以直接从内存中读取数据了，效率更高
        #这里root = root 不会报错 Python 完全能区分开“参数名”和“变量的值”，所以不会混淆，也不会报错。
        #datasets.MNIST 是torchvision库中提供的一个类，用于加载MNIST数据集。它会自动下载数据集（如果本地没有的话）并进行预处理。
        self.raw = datasets.MNIST(root = root, train= train, download=True)
        #存入transform属性 ，之后的方法可以调用 
        self.transform = transform

    # __len__方法是Dataset类的一个特殊方法，用于返回数据集的大小（即数据集中样本的数量）。当你使用len(dataset)时，Python会自动调用这个方法来获取数据集的长度。
    def __len__(self):
        #Python 的规则是：任何类，只要定义了 __len__ 方法，就可以用 len(实例) 来调用它
        return len(self.raw) 

    def __getitem__(self, idx):
        # 从内存中获取图像和标签
        image, label = self.raw[idx]
        # 如果定义了预处理方法，则应用它
        if self.transform:
            image = self.transform(image)
        # 返回处理后的图像和标签


        return image, label
    
    def get_dataloaders(self, batch_size=64, num_workers=0,train=True):
        #存储一个数据预处理方法，先转成Tensor格式（张量，理解为多维数组），再进行归一化处理（归一化的参数是先验的）
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))

        #使用MNISTDataset类创建一个实例，指定数据集的根目录、是否加载训练集以及数据预处理方法。
        data_set = MNISTDataset(root='./data', train=train, transform=transform)

        #
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    ])