import torch

def fgsm_attack(img,data_grad,epsilon):
    # data_grad=data.grad.data
    sign_data_grad=data_grad.sign()
    perturbed_img=img+epsilon*sign_data_grad
    perturbed_img=torch.clamp(perturbed_img,0,1)
    return perturbed_img