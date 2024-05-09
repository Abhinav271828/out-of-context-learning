from scripts.model_def import MultiheadAttention
import torch

def create_mha_construction(input_dim, num_heads, initial_weights, lr):
    """
    Create the weight construction for single-head attention
    to simulate gradient descent.
    We assume that the last element of the input is the label.
    """
    assert num_heads == 1, "This function is only for single-head attention!"

    w_q = torch.zeros((input_dim, input_dim))
    x_size = input_dim - 1
    w_q[range(x_size), range(x_size)] = 1

    w_k = w_q.clone()

    w_v = torch.zeros(input_dim, input_dim)
    w_v[-1:, :x_size] = initial_weights
    w_v[-1:, x_size:] = - torch.eye(1)

    p = (lr / num_heads) * torch.eye(input_dim)

    mha = MultiheadAttention(input_dim, input_dim, num_heads)

    mha.qkv_proj.weight.data = torch.concat([w_q, w_k, w_v], dim=1).transpose(0, 1)

    mha.o_proj.weight.data = p

    mha.requires_grad_(False)

    return mha