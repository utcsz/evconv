import torch
from dataloaders import mvsec_dataset
from models.fireflownet import get_model
from incr_models.fireflownetincr import get_incr_model
from tools.fireflow_utils import visualize_dataset, visualize_model, visualize_incr_model, time_model, time_incr_model
from tools.timing import cuda_timers
from preprocessors.fireflownet_preprocess import preprocess
import data_sources

if __name__ == '__main__':

    class config:
        data_root=data_sources.paths_config['mvsec']
        vdataset=False
        rec_lookback=5
        pretrained='./pretrained/fireflownet/FireFlowNet.pt'

    # load dataset and stuff
    dataset = mvsec_dataset.get_continuous_dataset(path=config.data_root)
    model = get_model().cuda()
    model.load_state_dict(torch.load('./pretrained/fireflownet/FireFlowNet.pt'))
    incr_model = get_incr_model().cuda()
    incr_model.load_state_dict(torch.load('./pretrained/fireflownet/FireFlowNet.pt'))

    # Let's load everything we need first
    PERFORM = "visualize incr"

    match PERFORM:
        case "visualize dataset":
            # let's look at the dataset shall we?
            visualize_dataset(dataset, preprocess, 1200)

        case "visualize baseline":
            # let's look at the output with baseline model
            visualize_model(model, dataset, 1000, preprocess, lambda x: x)

        case "visualize incr":
            # let's look at the output with baseline model
            visualize_incr_model(incr_model, dataset, 1000, preprocess, lambda x: x)

        case "time baseline":
            # let's time the baseline model
            time_model(model, dataset, 2, preprocess, lambda x: x)
            print(cuda_timers)


        case "time incr":
            # let's time the incremental model
            time_incr_model(incr_model, dataset, 2, preprocess, lambda x: x)
            print(cuda_timers)

        case _:
            print("Invalid PERFORM value")




