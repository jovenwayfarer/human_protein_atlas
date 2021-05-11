from main import HPALit
import torch
import glob
import tqdm

ckpt_models = glob.glob('logs/*/*.ckpt')
for i, checkpoint_path in enumerate(tqdm.tqdm(ckpt_models)):
    
    model = HPALit.load_from_checkpoint(checkpoint_path)
    model.eval()
    state_dict = model.state_dict()
    torch.save(state_dict, checkpoint_path.replace('ckpt','pth'))