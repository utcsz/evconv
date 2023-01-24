from .h5data.h5 import H5Loader

def get_continuous_dataset(path='/data/'):
    config = {'data': {'mode': 'time', 'num_bins': 5, 'path': path, 'window': 0.1}, 'experiment': 'Default', 'hot_filter': {'enabled': True, 'max_px': 100, 'max_rate': 0.8, 'min_obvs': 5}, 'loader': {'augment': [], 'augment_prob': [0.5, 0.5, 0.5], 'batch_size': 1, 'gpu': 0, 'n_epochs': 120, 'resolution': [256, 256], 'seed': 0}, 'loss': {'flow_regul_weight': 1}, 'model': {}, 'model_flow': {'base_num_channels': 32, 'kernel_size': 3, 'mask_output': True, 'name': 'FireFlowNetIncr', 'flow_scaling': 128}, 'optimizer': {'lr': 0.0001, 'name': 'Adam'}, 'prev_model': '', 'trained_model': 'trained_models/model_29072022_132353_92e71bd3d8244494b350b3add8a55b82/', 'vis': {'bars': False, 'enabled': True, 'px': 400, 'verbose': True, 'store': False}}
    dataset = H5Loader(config, config['data']['num_bins'])
    return dataset

