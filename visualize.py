# %%

import matplotlib.pyplot as plt
from scripts.transformer_with_auto_features import MembershipModelPlusAutoEncoderLightning

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
    qv_prod = q_k_v[1].T @ q_k_v[2]
    plot_a_matrix_heatmap(qv_prod)

    o_proj_v_prod = out_proj.T @ q_k_v[2]
    plot_a_matrix_heatmap(o_proj_v_prod)


# %%
transformer = send_multi_head_attn_from_membershipmodel("MGM/7wxdd8yu/checkpoints/epoch=32-step=41250.ckpt")
q_k_v, out_proj = multi_head_attn_layer_to_matrices(transformer)
plot_relevant_stuff(q_k_v, out_proj)

# %%

