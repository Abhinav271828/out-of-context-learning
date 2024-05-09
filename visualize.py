# %%

import matplotlib.pyplot as plt
from scripts.transformer_with_auto_features import MembershipModelPlusAutoEncoderLightning
from construction import create_mha_construction
import torch

def plot_a_matrix_heatmap(matrix):
    # make a heatmap for values ranging for -1 to 1 
    # 0 should be white, 1 should be blue and -1 is red
    plt.imshow(matrix, cmap='coolwarm')
    plt.colorbar()
    plt.clim(-1, 1)
    plt.show()

def send_multi_head_attn_from_membershipmodel(model_path):
    model = MembershipModelPlusAutoEncoderLightning.load_from_checkpoint(model_path)
    transformer = model.transformer.multihead_attn
    return transformer

def multi_head_attn_layer_to_matrices(multi_head_attn):
    qkv_proj = multi_head_attn.qkv_proj.weight.detach().cpu().numpy()
    out_proj = multi_head_attn.o_proj.weight.detach().cpu().numpy()
    q_k_v = qkv_proj.reshape(3, 17, 17)

    return q_k_v, out_proj

def plot_relevant_stuff(q_k_v, out_proj):
<<<<<<< Updated upstream
    kq_prod = q_k_v[1].T @ q_k_v[0]
    plot_a_matrix_heatmap(kq_prod)
=======
    qv_prod = q_k_v[1].T @ q_k_v[0]
    plot_a_matrix_heatmap(qv_prod)
>>>>>>> Stashed changes

    o_proj_v_prod = out_proj.T @ q_k_v[2]
    plot_a_matrix_heatmap(o_proj_v_prod)


# %%
<<<<<<< Updated upstream
mha = send_multi_head_attn_from_membershipmodel("MGM/7wxdd8yu/checkpoints/epoch=32-step=41250.ckpt")
plot_relevant_stuff(multi_head_attn_layer_to_matrices(mha))


mha = create_mha_construction(17, 1, torch.zeros(1, 16), 4)
plot_relevant_stuff(*multi_head_attn_layer_to_matrices(mha))
=======
transformer = send_multi_head_attn_from_membershipmodel("MGM/gyn51t8t/checkpoints/epoch=56-step=71250.ckpt")
q_k_v, out_proj = multi_head_attn_layer_to_matrices(transformer)
plot_relevant_stuff(q_k_v, out_proj)
>>>>>>> Stashed changes

# %%

