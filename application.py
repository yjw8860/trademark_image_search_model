import json
from search_ops import _main_

config_path = './config.json'
with open(config_path) as config_buffer:
    config = json.loads(config_buffer.read())

img_folder = config['application']['img_folder']
weight_path = config['application']['trained_weights']

if __name__=='__main__':
    _main_(config=config,
           save_feature=False,
           retrieval_images=False,
           calculate_MAP=True)
