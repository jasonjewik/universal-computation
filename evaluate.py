from universal_computation.fpt import FPT

model = FPT(input_dim=405, output_dim=16000)

# import torch
# import glob

# from universal_computation.fpt import FPT
# from universal_computation.datasets.deep_downscale import DeepDownscaleDataset

# mse_loss = torch.nn.MSELoss()

# def nan_to_num(t, mask=None):
#     if mask is None:
#         mask = torch.isnan(t)
#     zeros = torch.zeros_like(t)
#     return torch.where(mask, zeros, t)

# def acc(preds, true):
#     preds = preds[:, 0]
#     preds = nan_to_num(preds, torch.isnan(true))
#     true = nan_to_num(true)
#     return torch.sqrt(mse_loss(preds, true))

# model = FPT(input_dim=405, output_dim=16000)
# filepath = glob.glob('models/*.pt')[0]
# checkpoint = torch.load(filepath)
# model.load_state_dict(checkpoint['model'])
# model.eval()

# dset = DeepDownscaleDataset(batch_size=1, patch_size=3)
# x, y = dset.get_batch()

# preds = model(x)
# print(f'RMSE: {acc(preds, y).item()}')