import os
import os.path
import sys

restools_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(restools_root_path, 'thequickmath'))
sys.path.append(os.path.join(restools_root_path, 'pycomsdk'))
