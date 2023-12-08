import argparse 
from model import * 
import torch 
from loader import VideoDataset 
from glob import glob 
from tqdm import tqdm 

def main():
    model = SimVP((11, 3, 160, 240)) 
    model.load_state_dict(torch.load(args.model))

    model.eval() 

    data = VideoDataset(
        paths=glob(f'{args.data_path}/video_*'), 
        labeled=False, 
        video_len=11
    ) 
    loader = torch.utils.data.DataLoader(
            data,
            batch_size=args.readin_batch,
            num_workers=args.num_workers
    )

    res = [] 
    pbar = tqdm(loader, desc='Predicting', leave=False)
    for batch in pbar: 
        batch = batch.permute() # (B, T, H, W, C0 -> (B, T, C, H, W) 
        out = model(batch) # (B, T, C, H, W)  
        out = out[:, -1, :, :, :] # (B, C, H, W)  
        res.append(out)  
    res = torch.cat(res, dim=0) # (N, C, H, W) 
    
    print(res.shape) 

    torch.save(res, args.output_path) 

    print('Saved!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # get params 
    parser.add_argument("--data_path", type=str, help="input data path") 
    parser.add_argument("--output_path", type=str, help="output data path") 
    parser.add_argument("--model", type=str, help="model path for simvp") 
    parser.add_argument("--readin_batch", type=int, help="batch size for reading in/ loading data") 
    parser.add_argument("--num_workers", type=int, help="number of workers for processing the data") 

    args = parser.parse_args() 
    main(args) 