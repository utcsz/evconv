import torch
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
from dataloaders import dense_dataset
from models.e2depth import get_model
from incr_models.e2depthincr import get_incr_model
from preprocessors.e2depth_preprocess import preprocess
from tools.e2d_utils import time_model, time_incr_model, visualize_model, visualize_incr_model, visualize_dataset
from tools.timing import cuda_timers
import data_sources


if __name__ == '__main__':


    class config:
        data_root=data_sources.paths_config['dense']
        vdataset=False
        rec_lookback=50
        pretrained='./pretrained/e2depth/e2depth.pth.tar'
        

    # Let's load everything we need first
    dataset = dense_dataset.get_continuous_dataset(path=config.data_root, vdataset=config.vdataset)
    model = get_model().cuda()
    model.load_state_dict(torch.load(config.pretrained)['state_dict'])
    incr_model = get_incr_model().cuda()
    incr_model.load_state_dict(torch.load(config.pretrained)['state_dict'])

    PERFORM = "visualize incr"

    match PERFORM:
        case "visualize dataset":
            # let's look at the dataset shall we?
            visualize_dataset(dataset, preprocess, 1200)

        case "visualize baseline":
            # let's look at the output with baseline model
            visualize_model(model, dataset, 1000, preprocess, lambda x: x, timer_name='model_timer', rec_lookback=config.rec_lookback)

        case "visualize incr":
            # let's look at the output with baseline model
            # c = visualize_incr_model(incr_model, dataset, 200, preprocess, lambda x: x, rec_lookback=config.rec_lookback, refresh_only=True)
            c_incr = visualize_incr_model(incr_model, dataset, 200, preprocess, lambda x: x, timer_name='incr_timer', rec_lookback=config.rec_lookback, refresh_only=False)
            # print(c - c_incr)
            # print(torch.abs(c - c_incr).max())

        case "time baseline":
            # let's time the baseline model
            time_model(model, dataset, 2, preprocess, lambda x: x, rec_lookback=config.rec_lookback)
            print(cuda_timers)

        case "time incr":
            # let's time the incremental model
            time_incr_model(incr_model, dataset, 2, preprocess, lambda x: x,rec_lookback=config.rec_lookback)
            print(cuda_timers)

        case _:
            print("Invalid PERFORM value")





