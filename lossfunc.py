import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_ce(outs_t, outs_s_b1, outs_s_b2, labels, device):

    # 交叉熵损
    loss_cel = nn.CrossEntropyLoss()
    loss_cel_t = loss_cel(outs_t, labels.to(device))
    loss_cel_s_b1 = loss_cel(outs_s_b1, labels.to(device))
    loss_cel_s_b2 = loss_cel(outs_s_b2, labels.to(device))

    return loss_cel_t + loss_cel_s_b1 + loss_cel_s_b2


def loss_relation(outs_t, outs_s_b1, xt_vector, xt_map, xs_b2_map_last, xs_b2_vector_last, fc_b2_weights, device):

    # 不通过student模型参数更新来影响teacher模型
    xt_map_copy = xt_map.clone().detach()
    xt_vector_copy = xt_vector.clone().detach()

    # MSE损失
    loss_mse = nn.MSELoss()
    assert xt_map.shape == xs_b2_map_last.shape, 'dim is not matching!'
    xs_b2_map_cam_b2 = (xs_b2_map_last.unsqueeze(1) * fc_b2_weights).sum(2)
    xt_map_cam_b2 = (xt_map_copy.unsqueeze(1) * fc_b2_weights).sum(2)
    loss_mse_1 = loss_mse(xs_b2_map_cam_b2, xt_map_cam_b2)

    xs_b2_similar = torch.mm(xs_b2_vector_last, xs_b2_vector_last.t())
    xt_similar = torch.mm(xt_vector_copy, xt_vector_copy.t())
    xs_b2_similar_norm = F.normalize(xs_b2_similar, p=2, dim=1)
    xt_similar_norm = F.normalize(xt_similar, p=2, dim=1)
    loss_mse_2 = loss_mse(xs_b2_similar_norm, xt_similar_norm)

    loss_mse_3 = loss_mse(outs_t, outs_s_b1)
    return 1.2*loss_mse_1 + 2.0*loss_mse_2 + 2.0*loss_mse_3


def loss_kd(outs_t, outs_s_b2, device):

    outs_t_copy = outs_t.clone().detach()

    # 知识蒸馏
    t = 2.0
    loss_kld = nn.KLDivLoss(reduction='batchmean')
    log_softmax = nn.LogSoftmax(dim=1)
    soft_label = (outs_t_copy / t).softmax(1)
    loss_kd_2 = loss_kld(log_softmax(outs_s_b2 / t), soft_label) * t * t

    return 2.0*loss_kd_2
