import torch
from torch.utils.data import DataLoader
from .timing import CudaTimer
from .visualization import visualize_depth, visualize
import queue


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


def visualize_model(model, dataset, iterattions, preprocess, postprocess, timer_name='model_timer', rec_lookback=0):
    iters = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    hq = queue.Queue()
    with torch.no_grad():
        with CudaTimer(timer_name):
            for data in dataloader:
                if iters == iterattions:
                    break
                iters += 1

                data = preprocess(data)
                # no recurrent lookback
                if hq.qsize() >= rec_lookback:
                    data[1] = hq.get() 
                else:
                    data[1] = None
                c,h = model(data)
                hq.put(h)

                visualize(c[0], apply_colormap=False)


def visualize_incr_model(model, dataset, iterattions, preprocess, postprocess, timer_name='model_timer', rec_lookback=0, refresh_only=False):
    iters = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    hq = []
    hq_incr = queue.Queue()
    with torch.no_grad():
        for data in dataloader:
            if iters == iterattions:
                break
            iters += 1

            inp1 = preprocess(data)
            if refresh_only or iters%20 == 1:
                # baseline model
                if len(hq) >= rec_lookback:
                    h_1 = hq[-rec_lookback]
                    if len(hq) >= rec_lookback+2:
                        del hq[0]
                else:
                    h_1 = None

                inp = [inp1, h_1]
                c,h = model.forward_refresh_reservoirs(inp)
                hq.append(h)
                # inp1_prev = torch.zeros_like(inp)

            else:
                # incremental model

                # eventframe
                inp1_incr = [inp1 - inp1_prev, None]
                # input
                if len(hq) >= rec_lookback:
                    h_1 = hq[-rec_lookback]
                    if len(hq) >= rec_lookback+1:
                        h_0 = hq[-rec_lookback-1]
                        inp2_incr = [ [ [h_1[i][j] - h_0[i][j], None] for j in range(len(h[i])) ] for i in range(len(h))]
                    else:
                        inp2_incr = [ [ [h_1[i][j], None] for j in range(len(h[i])) ] for i in range(len(h))]

                    if len(hq) >= rec_lookback+4:
                        del hq[0]

                else:
                    inp2_incr = [ None for _ in range(model.num_encoders)]

                inp_incr = inp1_incr, inp2_incr
                with CudaTimer(timer_name):
                    c_incr, h_incr = model(inp_incr)
                c += c_incr[0]
                if h_incr[0] is not None:
                    h = [ [ h[i][j] + h_incr[i][j][0] for j in range(len(h[i])) ] for i in range(len(h))]
                hq.append(h)

    
            inp1_prev = inp1.clone().detach()

            visualize(c[0], apply_colormap=False)
            # print(c)
    return c



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
                c,h = model(data)
        print(c)




def time_incr_model(model, dataset, iterations, preprocess, postprocess, timer_name='model_timer', rec_lookback=0):
    iters = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    hq = []
    hq_incr = queue.Queue()
    with torch.no_grad():
        with CudaTimer(timer_name):
            for data in dataloader:
                if iters == iterations:
                    break
                iters += 1

                inp1 = preprocess(data)
                inp1_prev = inp1

                if iters%20 == 1:
                    # baseline model
                    if len(hq) >= rec_lookback:
                        h_1 = hq[-rec_lookback]
                        if len(hq) >= rec_lookback+2:
                            del hq[0]
                    else:
                        h_1 = None

                    inp1[1] = h_1
                    c,h = model.forward_refresh_reservoirs(inp1)
                    hq.append(h)

                else:
                    # incremental model

                    # eventframe
                    inp1_incr = [inp1[0] - inp1_prev[0], None]

                    # input
                    if len(hq) >= rec_lookback:
                        h_1 = hq[-rec_lookback]
                        if len(hq) >= rec_lookback+1:
                            h_0 = hq[-rec_lookback-1]
                            inp2_incr = [ [ [h_1[i][j] - h_0[i][j], None] for j in range(len(h[i])) ] for i in range(len(h))]
                        else:
                            inp2_incr = [ [ [h_1[i][j], None] for j in range(len(h[i])) ] for i in range(len(h))]

                        if len(hq) >= rec_lookback+4:
                            del hq[0]
                    else:
                        inp2_incr = [ None for _ in range(model.num_encoders)]

                    inp = inp1_incr, inp2_incr
                    c_incr, h_incr = model(inp)
                    c += c_incr[0]
                    h = [ [ h[i][j] + h_incr[i][j][0] for j in range(len(h[i])) ] for i in range(len(h))]

                    hq.append(h)




