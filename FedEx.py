# -*- coding: utf-8 -*-
import gfedplat as fp
import copy
import numpy as np
import torch
import time

class FedEx(fp.Algorithm):
    def __init__(self,
                 name='FedEx',
                 data_loader=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 client_num=None,
                 client_list=None,
                 online_client_num=None,
                 client_test=None,
                 max_comm_round=0,
                 max_training_num=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 write_log=True,
                 dishonest=None,
                 params=None,
                 *args,
                 **kwargs):

        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay'])
        # 调用父类构造方法完成实例化
        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num, client_test=client_test, max_comm_round=max_comm_round, max_training_num=max_training_num, epochs=epochs, save_name=save_name, outFunc=outFunc, write_log=write_log, dishonest=dishonest, params=params, *args, **kwargs)
        # 定义一个历史信息记录器
        self.client_online_round_history = [None] * self.client_num  # 记录每个client上一次在线是哪一代
        self.client_gradient_history = [None] * self.client_num  # 记录每个client的最后一次在线的梯度

        self.used_history_flag = False  # 记录是否用了历史梯度

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))

    def run(self):
        print("FedEx run method started")  # Diagnostic print to confirm run is called
        batch_num = np.mean(self.get_clinet_attr('training_batch_num'))
        while not self.terminated():
            com_time_start = time.time()

            m_locals, _ = self.train_a_round()
            com_time_end = time.time()
            cal_time_start = time.time()
            old_model = self.module.span_model_params_to_vec()

            self.weight_aggregate_fairness(m_locals)

            self.current_training_num += self.epochs * batch_num

            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start

            print(f"Online client list length: {len(self.online_client_list)}")  # Diagnostic print for client list size

    def weight_aggregate_fairness(self, m_locals):
        """
        Weighted aggregation based on participation frequency and cosine similarity.
        Args:
            m_locals: list of local model objects
        """
        eps = 1e-8
        k = self.params.get('fairness_scaling_k', 50)
        fairness_exponent_tau = self.params.get('fairness_exponent_tau', 0.5)  # exponent scaling factor for fairness weighting

        # Compute dynamic sliding window tau based on online client history (same logic as in train_a_round)
        total_client_num = 0
        for item in self.client_online_round_history:
            if item is not None:
                total_client_num += 1
        if total_client_num > self.online_client_num:
            sliding_tau = int(total_client_num / self.online_client_num)
        else:
            sliding_tau = 1  # fallback to 1 if condition not met

        # Compute participation frequency fi for each client over sliding window sliding_tau
        fi_list = []
        for client_id in range(self.client_num):
            last_online_round = self.client_online_round_history[client_id]
            if last_online_round is None:
                fi = 0.0
            else:
                # Count how many rounds client was online in last sliding_tau rounds
                count = 0
                for r in range(self.current_comm_round - sliding_tau + 1, self.current_comm_round + 1):
                    if self.client_online_round_history[client_id] is not None and self.client_online_round_history[client_id] >= r:
                        count += 1
                fi = count / sliding_tau
            fi_list.append(fi)

        # Filter fi_list to only include online clients
        online_client_ids = [client.id for client in self.online_client_list]
        fi_list_online = [fi_list[client_id] for client_id in online_client_ids]

        fi_tensor = torch.tensor(fi_list_online, dtype=torch.float32, device=self.device)

        # Remove usage of self.alignment_matrix and always set alpha to ones
        alpha = torch.ones(len(online_client_ids), device=self.device)

        # Compute weights wi = (1/(fi + eps))^fairness_exponent_tau * sigmoid(k * alpha_i)
        weights = ((1.0 / (fi_tensor + eps)) ** fairness_exponent_tau) * self.sigmoid(k * alpha)

        # Normalize weights
        weights = weights / weights.sum()

        # Extract parameter dicts from model objects or use state_dicts directly
        state_dicts = []
        for model in m_locals:
            if hasattr(model, 'named_parameters') and callable(getattr(model, 'named_parameters')):
                param_dict = {}
                for name, param in model.named_parameters():
                    param_dict[name] = param.data.clone()
                state_dicts.append(param_dict)
            elif hasattr(model, 'state_dict') and callable(getattr(model, 'state_dict')):
                # Use state_dict method to get parameters
                state_dicts.append(model.state_dict())
            elif hasattr(model, 'model'):
                # Check if model has .model attribute with named_parameters or state_dict
                inner_model = model.model
                if hasattr(inner_model, 'named_parameters') and callable(getattr(inner_model, 'named_parameters')):
                    param_dict = {}
                    for name, param in inner_model.named_parameters():
                        param_dict[name] = param.data.clone()
                    state_dicts.append(param_dict)
                elif hasattr(inner_model, 'state_dict') and callable(getattr(inner_model, 'state_dict')):
                    state_dicts.append(inner_model.state_dict())
                else:
                    raise AttributeError(f"Inner model object {inner_model} has no named_parameters method or state_dict method")
            elif isinstance(model, dict):
                # Assume model is a state_dict
                state_dicts.append(model)
            else:
                raise AttributeError(f"Model object {model} has no named_parameters method, no state_dict method, and is not a state_dict")

        # Weighted aggregation of parameter dicts
        aggregated = copy.deepcopy(state_dicts[0])

        # Filter keys to only tensor values
        tensor_keys = [key for key, value in aggregated.items() if torch.is_tensor(value)]

        for key in tensor_keys:
            aggregated[key] = torch.zeros_like(aggregated[key])

        for i, local_params in enumerate(state_dicts):
            for key in tensor_keys:
                aggregated[key] += weights[i] * local_params[key]

        # Update model parameters with aggregated result
        if hasattr(self.module, 'model') and hasattr(self.module.model, 'load_state_dict'):
            self.module.model.load_state_dict(aggregated)
        else:
            self.module.load_state_dict(aggregated)

    def train_a_round(self):
        """
        进行一轮训练。
        """
        print("FedEx train_a_round called")  # Unique print to confirm updated code execution

        # Reset accuracy sums and counts for the current round to avoid accumulation across rounds
        round_num = self.current_comm_round

        com_time_start = time.time()

        # Call parent train() to get model objects and losses
        m_locals, l_locals = super().train()

        com_time_end = time.time()

        # Evaluate gradients and losses for custom logic
        g_locals, l_locals_eval = self.evaluate()

        # 处理historical公平性
        client_id_list = self.get_clinet_attr('id')  # 获取当前个体的id
        add_grads = []
        self.used_history_flag = False
        # 先计算历史上线个体总数
        total_client_num = 0
        for item in self.client_online_round_history:
            if item is not None:
                total_client_num += 1
        if total_client_num > self.online_client_num:  # 排除total_client_num = online个体数的情况，因为此时并没有用户掉线
            tau = int(total_client_num / self.online_client_num)
            for client_id, item in enumerate(self.client_online_round_history):
                if item is not None:
                    if self.current_comm_round - item <= tau:  # 例如当前t=100, tau=10, 则考虑91-99代的历史用户的梯度
                        if client_id not in client_id_list:  # 排除当代在线的用户
                            add_grads.append(self.client_gradient_history[client_id])
        if len(add_grads) == 0:
            add_grads = None
        else:
            add_grads = torch.vstack(add_grads)
            self.used_history_flag = True

        # 创建偏好向量
        prefer_vec = torch.Tensor([1.0] * self.online_client_num).float().to(self.device)
        prefer_vec = prefer_vec / torch.norm(prefer_vec)

        # Removed layer-wise direction calculation and QP step

        # Removed knowledge distillation and alignment matrix calculation block

        # Perform weighted fairness aggregation instead of FedAvg aggregation
        self.weight_aggregate_fairness(m_locals)

        # Perform client testing (accuracy calculation) after aggregation
        for client in self.online_client_list:
            client.test(self.module)

        self.current_training_num += 1

        last_client_id_list = self.get_clinet_attr('id')
        last_g_locals = copy.deepcopy(g_locals)
        for idx, client_id in enumerate(last_client_id_list):
            self.client_online_round_history[client_id] = self.current_comm_round
            temp = self.client_gradient_history[client_id]
            self.client_gradient_history[client_id] = None
            del temp
            self.client_gradient_history[client_id] = last_g_locals[idx]

        self.communication_time += com_time_end - com_time_start

        return m_locals, l_locals
