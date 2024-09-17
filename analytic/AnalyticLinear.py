# -*- coding: utf-8 -*-
"""
Basic analytic linear modules for the analytic continual learning [1-5].

References:
[1] Zhuang, Huiping, et al.
    "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
    Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
[2] Zhuang, Huiping, et al.
    "GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
[3] Zhuang, Huiping, et al.
    "DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning."
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.
[4] Zhuang, Huiping, et al.
    "G-ACIL: Analytic Learning for Exemplar-Free Generalized Class Incremental Learning"
    arXiv preprint arXiv:2403.15706 (2024).
[5] Fang, Di, et al.
    "AIR: Analytic Imbalance Rectifier for Continual Learning."
    arXiv preprint arXiv:2408.10349 (2024).
"""

import torch
from torch.nn import functional as F
from typing import Optional, Union
from abc import abstractmethod, ABCMeta


class AnalyticLinear(torch.nn.Linear, metaclass=ABCMeta):
    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super(torch.nn.Linear, self).__init__()  # Skip the Linear class
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gamma: float = gamma
        self.bias: bool = bias
        self.dtype = dtype

        # Linear Layer
        if bias:
            in_features += 1
        weight = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

    @torch.inference_mode()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight)
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)
        return X @ self.weight

    @property
    def in_features(self) -> int:
        if self.bias:
            return self.weight.shape[0] - 1
        return self.weight.shape[0]

    @property
    def out_features(self) -> int:
        return self.weight.shape[1]

    def reset_parameters(self) -> None:
        # Following the equation (4) of ACIL, self.weight is set to \hat{W}_{FCN}^{-1}
        self.weight = torch.zeros((self.weight.shape[0], 0)).to(self.weight)

    @abstractmethod
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        raise NotImplementedError()

    def update(self) -> None:
        assert torch.isfinite(self.weight).all(), (
            "Pay attention to the numerical stability! "
            "A possible solution is to increase the value of gamma. "
            "Setting self.dtype=torch.double also helps."
        )


class RecursiveLinear(AnalyticLinear):
    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super().__init__(in_features, gamma, bias, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        # Regularized Feature Autocorrelation Matrix (RFAuM)
        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """The core code of the ACIL and the G-ACIL.
        This implementation, which is different but equivalent to the equations shown in [1],
        is proposed in the G-ACIL [4], which supports mini-batch learning and the general CIL setting.
        """
        X, Y = X.to(self.weight), Y.to(self.weight)
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)

        num_targets = Y.shape[1]
        if num_targets > self.out_features:
            increment_size = num_targets - self.out_features
            tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
            self.weight = torch.cat((self.weight, tail), dim=1)
        elif num_targets < self.out_features:
            increment_size = self.out_features - num_targets
            tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
            Y = torch.cat((Y, tail), dim=1)

        # Please update your PyTorch & CUDA if the `cusolver error` occurs.
        # If you insist on using this version, doing the `torch.inverse` on CPUs might help.
        # >>> K_inv = torch.eye(X.shape[0]).to(X) + X @ self.R @ X.T
        # >>> K = torch.inverse(K_inv.cpu()).to(self.weight.device)
        K = torch.inverse(torch.eye(X.shape[0]).to(X) + X @ self.R @ X.T)
        # Equation (10) of ACIL
        self.R -= self.R @ X.T @ K @ X @ self.R
        # Equation (9) of ACIL
        self.weight += self.R @ X.T @ (Y - X @ self.weight)


class GeneralizedARM(AnalyticLinear):
    """Analytic Re-weighting Module (ARM) for generalized CIL."""

    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:    
        # print(gamma)
        super().__init__(in_features, gamma, bias, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        weight = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

        A = torch.zeros((0, in_features, in_features), **factory_kwargs)
        self.register_buffer("A", A)

        C = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("C", C)

        self.cnt = torch.zeros(0, dtype=torch.int, device=device)

    @property
    def out_features(self) -> int:
        return self.C.shape[1]

    @torch.inference_mode()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self.weight)
        # Bias
        if self.bias:
           
            X = torch.concat((X, torch.ones(X.shape[0], 1)), dim=-1)

        # GCIL
        num_targets = int(y.max()) + 1
        if num_targets > self.out_features:
            increment_size = num_targets - self.out_features
            torch.cuda.empty_cache()
            # Increment C
            tail = torch.zeros((self.C.shape[0], increment_size)).to(self.weight)
            self.C = torch.concat((self.C, tail), dim=1)
            # Increment cnt
            tail = torch.zeros((increment_size,)).to(self.cnt)
            self.cnt = torch.concat((self.cnt, tail))
            # Increment A

            tail = torch.zeros((increment_size, self.in_features, self.in_features))
            self.A = torch.concat((self.A, tail.to(self.A)))
            torch.cuda.empty_cache()
        else:
            num_targets = self.out_features

        # ACIL
        Y = F.one_hot(y, max(num_targets, num_targets)).to(self.C)
        self.C += X.T @ Y

        # Label Balancing
        y_labels, label_cnt = torch.unique(y, sorted=True, return_counts=True)
        y_labels, label_cnt = y_labels.to(self.cnt.device), label_cnt.to(
            self.cnt.device
        )
        self.cnt[y_labels] += label_cnt

        # Accumulate
        for i in range(num_targets):
            X_i = X[y == i]
            self.A[i] += X_i.T @ X_i

    @torch.inference_mode()
    def update(self):
        cnt_inv = 1 / self.cnt.to(self.dtype)
        cnt_inv[torch.isinf(cnt_inv)] = 0  # replace inf with 0
        cnt_inv *= len(self.cnt) / cnt_inv.sum()


        weighted_A = torch.sum(cnt_inv[:, None, None].mul(self.A), dim=0)
        A = weighted_A + self.gamma * torch.eye(self.in_features).to(self.A)
        C = self.C.mul(cnt_inv[None, :])

        self.weight = torch.inverse(A) @ C


class GeneralizedARM2(AnalyticLinear):
    """Analytic Re-weighting Module (ARM) for generalized CIL."""

    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
      
        # print(gamma)
        super().__init__(in_features, gamma, bias, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        weight = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

        A = torch.zeros((0, in_features, in_features), **factory_kwargs)
        self.register_buffer("A", A)

        C = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("C", C)

        self.cnt = torch.zeros(0, dtype=torch.int, device=device)

    @property
    def out_features(self) -> int:
        return self.C.shape[1]

    @torch.inference_mode()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        X = X.to(self.weight)
        # Bias
        if self.bias:
          
            X = torch.concat((X, torch.ones(X.shape[0], 1)), dim=-1)

        # GCIL
        num_targets = Y.shape[1]
        if num_targets > self.out_features:
            increment_size = num_targets - self.out_features
            torch.cuda.empty_cache()
            # Increment C
            tail = torch.zeros((self.C.shape[0], increment_size)).to(self.weight)
            self.C = torch.concat((self.C, tail), dim=1)
            # Increment cnt
            tail = torch.zeros((increment_size,)).to(self.cnt)
            self.cnt = torch.concat((self.cnt, tail))
            # Increment A

            tail = torch.zeros((increment_size, self.in_features, self.in_features))
            self.A = torch.concat((self.A, tail.to(self.A)))
            torch.cuda.empty_cache()
        else:
            num_targets = self.out_features

        # 将 Y 转换为与 X 相同的 dtype
        Y = Y.to(self.dtype)
        # ACIL
        self.C += X.T @ Y
        labels = torch.argmax(Y, dim=1)  # 获取批次中的类别索引

        # Label Balancing
        y_labels, label_cnt = torch.unique(labels, sorted=True, return_counts=True)
        y_labels, label_cnt = y_labels.to(self.cnt.device), label_cnt.to(
            self.cnt.device
        )
        self.cnt[y_labels] += label_cnt

        # Accumulate
        for i in range(num_targets):
            X_i = X[labels == i]
            self.A[i] += X_i.T @ X_i

    @torch.inference_mode()
    def update(self):
        cnt_inv = 1 / self.cnt.to(self.dtype)
        cnt_inv[torch.isinf(cnt_inv)] = 0  # replace inf with 0
        cnt_inv *= len(self.cnt) / cnt_inv.sum()

        weighted_A = torch.sum(cnt_inv[:, None, None].mul(self.A), dim=0)
        A = weighted_A + self.gamma * torch.eye(self.in_features).to(self.A)
        C = self.C.mul(cnt_inv[None, :])

        self.weight = torch.inverse(A) @ C



class GeneralizedDSIR(AnalyticLinear):
    """Analytic Re-weighting Module (ARM) with Compensation Stream for generalized CIL."""

    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
        activation_main: Optional[torch.nn.Module] = torch.relu,  # 主流激活函数
        activation_comp: Optional[torch.nn.Module] = torch.tanh,  # 补偿流激活函数 
    ) -> None:

        
        super().__init__(in_features, gamma, bias, device, dtype)

        factory_kwargs = {"device": device, "dtype": dtype}
        # 主流权重初始化
        weight = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

        # 主流自相关矩阵和交叉相关矩阵
        A = torch.zeros((0, in_features, in_features), **factory_kwargs)
        self.register_buffer("A", A)

        C = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("C", C)

        # 补偿流自相关矩阵和交叉相关矩阵
        A_comp = torch.zeros((0, in_features, in_features), **factory_kwargs)
        self.register_buffer("A_comp", A_comp)

        C_comp = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("C_comp", C_comp)

        self.cnt = torch.zeros(0, dtype=torch.int, device=device)
        self.alpha = 0.6 # 补偿流的权重比例

        # 激活函数
        self.activation_main = activation_main
        self.activation_comp = activation_comp

    @property
    def out_features(self) -> int:
        return self.C.shape[1]

    @torch.inference_mode()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self.weight)
        if self.bias:
          
            X = torch.concat((X, torch.ones(X.shape[0], 1, device=X.device)), dim=-1)


        # 使用主流的激活函数
        X_main = self.activation_main(X)

        # GCIL 主流更新
        num_targets = int(y.max()) + 1

        # 确保 weight 的列数与类别数量匹配
        if self.weight.shape[1] < num_targets:
            new_weight = torch.zeros((self.weight.shape[0], num_targets), device=self.weight.device, dtype=self.weight.dtype)
            new_weight[:, :self.weight.shape[1]] = self.weight
            self.weight = new_weight

        # 如果有新类别被引入，则定义increment_size
        if num_targets > self.out_features:
            increment_size = num_targets - self.out_features  # 新类别的数量
            torch.cuda.empty_cache()
            # Increment C and A for main stream
            self.C = torch.concat((self.C, torch.zeros((self.C.shape[0], increment_size)).to(self.weight)), dim=1)
            self.A = torch.concat((self.A, torch.zeros((increment_size, self.in_features, self.in_features)).to(self.A)))
            self.C_comp = torch.concat((self.C_comp, torch.zeros((self.C_comp.shape[0], increment_size)).to(self.weight)), dim=1)
            self.A_comp = torch.concat((self.A_comp, torch.zeros((increment_size, self.in_features, self.in_features)).to(self.A_comp)))
            self.cnt = torch.concat((self.cnt, torch.zeros((increment_size,)).to(self.cnt)))
            torch.cuda.empty_cache()
        else:
            increment_size = 0 

        # ACIL 主流更新
        Y_main = F.one_hot(y, num_classes=num_targets).to(self.C)
        self.C += X_main.T @ Y_main

        # Label Balancing 主流部分
        y_labels, label_cnt = torch.unique(y, sorted=True, return_counts=True)
        y_labels, label_cnt = y_labels.to(self.cnt.device), label_cnt.to(self.cnt.device)
        self.cnt[y_labels] += label_cnt

        # 累积主流自相关矩阵
        for i in range(num_targets):
            X_i = X_main[y == i]
            self.A[i] += X_i.T @ X_i


        # 计算主流的残差 (用于补偿流)
        with torch.no_grad():
            Y_pred = X_main @ self.weight
            Y_one_hot = F.one_hot(y, num_classes=Y_pred.shape[1]).to(Y_pred.dtype)
            residual = Y_one_hot - Y_pred  # 计算残差


        # 如果有增量类别，PLC步骤：将旧类别的残差清零
        
        residual[:, :-increment_size] = 0 

        # 使用补偿流的激活函数
        X_comp = self.activation_comp(X)

        # 补偿流更新
        self.C_comp += X_comp.T @ residual
        for i in range(num_targets):
            X_i = X_comp[y == i]
            self.A_comp[i] += X_i.T @ X_i


    @torch.inference_mode()
    def update(self):
        # 主流权重更新
        cnt_inv = 1 / self.cnt.to(self.dtype)
        cnt_inv[torch.isinf(cnt_inv)] = 0  # 将无穷大的部分设为0
        cnt_inv[cnt_inv < 1e-10] = 1e-10  # 防止极小值影响稳定性
        cnt_inv *= len(self.cnt) / cnt_inv.sum()

        weighted_A = torch.sum(cnt_inv[:, None, None].mul(self.A), dim=0)
        A = weighted_A + self.gamma * torch.eye(self.in_features).to(self.A)
        C = self.C.mul(cnt_inv[None, :])

        # 解析解更新主流权重
        self.weight = torch.inverse(A) @ C

        # 补偿流权重更新
        weighted_A_comp = torch.sum(cnt_inv[:, None, None].mul(self.A_comp), dim=0)
        A_comp = weighted_A_comp + self.gamma * torch.eye(self.in_features).to(self.A_comp)
        C_comp = self.C_comp.mul(cnt_inv[None, :])

        # 解析解更新补偿流权重
        weight_comp = torch.inverse(A_comp) @ C_comp

        # 最终权重为主流和补偿流的加权和
        self.weight = self.weight + self.alpha * weight_comp



