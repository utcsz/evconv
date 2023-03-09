import torch
from torch.utils.data import DataLoader
from .timing import CudaTimer
from .visualization import visualize


def visualize_dataset(dataset, preprocess, iterations):
    iters = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    h = None
    for data in dataloader:
        if iters == iterations:
            break
        iters += 1
        data = preprocess(data)
        visualize(data)


def visualize_model(model, dataset, iterattions, preprocess, postprocess, timer_name='model_timer'):
    iters = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    h = None
    with torch.no_grad():
        with CudaTimer(timer_name):
            for data in dataloader:
                if iters == iterattions:
                    break
                iters += 1
                data = preprocess(data)
                c = model(data)
                print(c)

def visualize_incr_model(model, dataset, iterattions, preprocess, postprocess, timer_name='model_timer'):
    return NotImplementedError('visualize_incr_model(): Not implemented yet')

def time_model(model, dataset, iterattions, preprocess, postprocess, timer_name='model_timer'):
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
 

def time_incr_model(model, dataset, iterations, preprocess, postprocess, timer_name='model_timer'):
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
                    inp = [data[0], None]
                    out = model(inp)





