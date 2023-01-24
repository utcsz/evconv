import torch
from torch.utils.data import DataLoader
from .timing import CudaTimer

def run_model_over_dataset_sequentially(model, dataset, iterattions, preprocess, postprocess, timer_name='model_timer'):
    iters = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    with torch.no_grad():
        with CudaTimer(timer_name):
            for data in dataloader:
                if iters == iterattions:
                    break
                iters += 1
                data = preprocess(data)
                out = model(data)
 

def run_incr_model_over_dataset_sequentially(model, dataset, iterations, preprocess, postprocess, timer_name='model_timer'):
    iters = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    with torch.no_grad():
        with CudaTimer(timer_name):
            for data in dataloader:
                if iters == iterations:
                    break
                iters += 1

                data = preprocess(data)
                data = data
                if iters%20 == 1:
                    out = model.forward_refresh_reservoirs(data)
                else:
                    inp = [data[0], None], None
                    c_incr, h_incr = model(inp)




def run_rec_model_over_dataset_sequentially(model, dataset, iterattions, preprocess, postprocess, timer_name='model_timer'):
    iters = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    with torch.no_grad():
        with CudaTimer(timer_name):
            for data in dataloader:
                if iters == iterattions:
                    break
                iters += 1
                data = preprocess(data)
                inp = [data, None]
                out = model(data)
                postprocess(out)        


