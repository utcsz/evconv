import torch
from torch.utils.data import DataLoader
from .timing import CudaTimer
from .visualization import visualize_flow, visualize


def visualize_dataset(dataset, preprocess, iterations):
    iters = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    h = None
    for data in dataloader:
        if iters == iterations:
            break
        iters += 1
        data = preprocess(data)
        visualize(data[0])


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
                flow = model(data)
                visualize_flow(flow)


def visualize_incr_model(model, dataset, iterations, preprocess, postprocess, timer_name='model_timer'):
    iters = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    with torch.no_grad():
        for data in dataloader:
            if iters == iterations:
                break
            iters += 1
            data = preprocess(data)
            if iters%20 == 1:
                out = model.forward_refresh_reservoirs(data)
                flow = out
            else:
                inp = [data-data_prev, None]
                out = model(inp)
                flow += out[0]

            data_prev = data

            visualize_flow(flow)




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





