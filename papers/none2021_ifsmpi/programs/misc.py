import os
import sys
import json

import numpy as np


def upload_paths_from_config():
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f:
        conf = json.load(f)
    sys.path.append(conf['path_to_thequickmath.reduced_models'])
    sys.path.append(conf['path_to_pyESN'])
