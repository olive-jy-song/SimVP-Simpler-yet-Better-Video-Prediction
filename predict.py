import argparse 
from model import * 
import torch 
from loader import VideoDataset 
from glob import glob 
from tqdm import tqdm 

def main(args):
    model = SimVP(args.in_shape) 
    model.load_state_dict(torch.load(args.model))

    model.eval() 

    data = VideoDataset(
        paths=glob(f'{args.data_path}/video_*'), 
        train=False, 
        image=args.pred_img, 
        video_len=11
    ) 
    loader = torch.utils.data.DataLoader(
            data,
            batch_size=args.readin_batch,
            num_workers=args.num_workers
    )

    res = [] 
    pbar = tqdm(loader, desc='Predicting', leave=False)
    for i, batch in enumerate(pbar): 
        if args.pred_img: 
            batch = batch.permute(0, 1, 4, 2, 3) # (B, T, H, W, C) -> (B, T, C, H, W) 
        out = model(batch) # (B, T, C, H, W)  
        out = out[:, -1, :, :, :] # (B, C, H, W)  
        res.append(out)  
        if args.pred_img and i == 3: 
            break 
    res = torch.cat(res, dim=0) # (N, C, H, W) 
    print('shape is ', res.shape) 

    if not args.pred_img:
        res = res.squeeze(1) 
        print('shape is ', res.shape) 

    torch.save(res, args.output_path) 

    print('Saved!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # get params 
    parser.add_argument("--data_path", type=str, help="input data path") 
    parser.add_argument("--output_path", type=str, help="output folder path") 
    parser.add_argument("--model", type=str, help="model path for simvp") 
    parser.add_argument("--readin_batch", type=int, default=16, help="batch size for reading in/ loading data") 
    parser.add_argument("--num_workers", type=int, default=1, help="number of workers for processing the data") 
    parser.add_argument('--pred_img', action='store_true', help='training image or training mask')
    parser.add_argument('--in_shape', default=[11,3,160,240], type=int,nargs='*') 


    args = parser.parse_args() 
    main(args) 

