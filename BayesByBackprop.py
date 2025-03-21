from torch.distributions import Normal
from torch import nn
from torch.nn import functional as F
import torch
import math


class Prior:
    """先验分布（式38），两个均值为0，方差不同的混合高斯分布"""

    def __init__(self, sigma1=1, sigma2=0.00001, pi=0.5):
        """
        Args:
        :param sigma1: 式38中的sigma1
        :param sigma2: 式38中的sigma2
        :param pi: 式38中的pi
        """
        self.normal1 = Normal(0, sigma1)
        self.normal2 = Normal(0, sigma2)
        self.pi = pi

    def log_prob(self, inputs):
        """
        计算先验分布的对数概率之和。因为各输入满足独立同分布，联合分布即为各分布之积，取对数变为各分布之和
        对数先验分布用于计算变分自由能
        """
        prob1 = self.normal1.log_prob(inputs).exp()  # 高斯分布1的概率密度函数
        prob2 = self.normal2.log_prob(inputs).exp()  # 高斯分布2的概率密度函数
        return (self.pi * prob1 + (1 - self.pi) * prob2).log().sum()


class VariationalPoster:
    """变分后验分布"""

    def __init__(self):
        self.normal = Normal(0, 1)
        self.sigma = None

    def sample(self, mu, rho):
        """
        重参数化采样
        从标准正态分布中采样epsilon，通过变换得到θ=μ+log(1+exp(ρ))·epsilon
        """
        self.mu = mu
        self.sigma = rho.exp().log1p()
        epsilon = self.normal.sample(mu.shape).to(mu.device)  # 算法2:第5行
        return self.mu + self.sigma * epsilon  # 式33            算法2:第6行

    def log_prob(self, inputs):
        """
        正态分布的概率密度
        log(N(x|mu, sigma)) = -log(sqrt(2*pi))-log(sigma) - (x-mu)^2 / (2*sigma^2)
        参数θ服从变分后验分布(即对角高斯分布)，epsilon服从标准正态分布
        将epsilon代入参数θ的概率密度函数，化简得到θ的概率密度函数就等于epsilon标准正态分布的概率密度函数
        """
        return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma)
                - ((inputs - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class BayesLinear(nn.Module):
    """贝叶斯线性层"""

    def __init__(self, in_features, out_features, prior):
        """
        Args:
        :param in_features: 输入维度
        :param out_features: 输出维度
        :param prior: 先验分布
               mu:    μ
               rho:   ρ
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.W_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))

        self.b_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.b_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        self.prior = prior
        self.W_variational_post = VariationalPoster()
        self.b_variational_post = VariationalPoster()

    def sample_weight(self):
        """从变分后验分布中采样全连接层的权重矩阵和偏置向量"""
        W = self.W_variational_post.sample(self.W_mu, self.W_rho)
        b = self.b_variational_post.sample(self.b_mu, self.b_rho)
        return W, b

    def forward(self, inputs, train=True):
        W, b = self.sample_weight()  # 采样权重矩阵和偏置向量
        outputs = F.linear(inputs, W.to(inputs.device), b.to(inputs.device))  # outputs = Wx + b

        # 预测
        if not train:
            return outputs, 0, 0

        # 训练
        # 对数先验分布
        log_prior = self.prior.log_prob(W).sum() + self.prior.log_prob(b).sum()  # 算法2: 第7行
        # 对数变分后验分布
        log_va_poster = self.W_variational_post.log_prob(W) + self.b_variational_post.log_prob(b)  # 算法2: 第7行
        return outputs, log_prior, log_va_poster


class BayesMLP(nn.Module):
    """贝叶斯MLP模型"""

    def __init__(self, in_dim, out_dim, hidden_dims, sigma1=1, sigma2=0.00001, pi=0.5, activate='none'):
        super().__init__()
        prior = Prior(sigma1, sigma2, pi)
        self.layers = nn.ModuleList()
        for dim in hidden_dims:
            self.layers.append(BayesLinear(in_dim, dim, prior))
            in_dim = dim
        self.layers.append(BayesLinear(in_dim, out_dim, prior))

        self.act_fn = F.tanh
        if activate == 'relu':
            self.act_fn = F.relu
        elif activate == 'sigmoid':
            self.act_fn = F.sigmoid
        self.flatten = nn.Flatten()
        self.training = True

    def run_sample(self, inputs, train):
        """执行一次采样，返回模型预测值、对数先验分布和对数变分后验分布"""
        if len(inputs.shape) >= 3:  # 样本是矩阵而不是向量的情况（例如图像）
            inputs = self.flatten(inputs)
        log_prior, log_va_poster = 0, 0  # 对数先验分布， 对数变分后验分布
        for layer in self.layers:
            model_preds, layer_log_prior, layer_log_va_poster = layer(inputs, train)
            log_prior += layer_log_prior
            log_va_poster += layer_log_va_poster
            inputs = self.act_fn(model_preds)

        return model_preds, log_prior, log_va_poster

    def forward(self, inputs, sample_num):
        """
        进行sample_num次蒙特卡洛采样，用于估计变分自由能
        Args
        :param inputs: 模型参数
        :param sample_num: 采样次数(式29中的m)
        """
        log_prior_s = 0
        log_va_poster_s = 0
        model_preds_s = []

        for _ in range(sample_num):  # 算法2:第4行
            model_preds, log_prior, log_va_poster = self.run_sample(inputs, self.training)
            log_prior_s += log_prior  # 对数先验分布
            log_va_poster_s += log_va_poster  # 对数变分后验分布
            model_preds_s.append(model_preds)  # 模型预测

        if not self.training:
            return model_preds_s
        else:
            return model_preds_s, log_prior_s, log_va_poster_s


class RegressionELBOLoss(nn.Module):
    """用于回归问题的ELBO证据下界损失"""

    def __init__(self, batch_num, noise_tol=0.1):
        super().__init__()
        self.batch_num = batch_num
        self.noise_tol = noise_tol

    def forward(self, model_out, targets):
        model_preds_s, log_prior, log_va_poster = model_out  # 输出
        # 对数似然
        log_like_s = 0
        for model_preds in model_preds_s:  # 算法2:第7行第3部分
            # 回归问题中的模型输入被认为是以预测结果为均值的高斯分布
            # 在这种情况下，单次采样的单个似然项不再满足独立同分布，只能蒙特卡洛采样，故返回的时候要除以len
            dist = Normal(model_preds, self.noise_tol)
            log_like_s += dist.log_prob(targets).sum()

        # 返回变分自由能作为损失
        return 1 / self.batch_num * (log_va_poster - log_prior) - log_like_s / len(model_preds_s)  # 算法2: 第8行


class ClassificationELBOLoss(nn.Module):
    """用于分类问题的ELBO证据下界损失"""

    def __init__(self, batch_num):
        super().__init__()
        self.batch_num = batch_num

    def forward(self, model_out, targets):
        model_preds_s, log_prior, log_va_poster = model_out
        neg_log_like_s = 0
        for model_preds in model_preds_s:
            # 每个采样样本的交叉熵就是他的期望负对数似然
            neg_log_like_s += F.cross_entropy(model_preds, targets, reduction='sum')

        # 返回变分自由能作为损失
        return 1 / self.batch_num * (log_va_poster - log_prior) + neg_log_like_s / len(model_preds_s)
