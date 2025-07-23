'''
Author: Zhongming Liang lzzzmmgpt@gmail.com
Date: 2025-07-23 10:15:11
LastEditors: Zhongming Liang lzzzmmgpt@gmail.com
LastEditTime: 2025-07-23 10:16:43
FilePath: /530/CellNiche/release/cellniche/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

__version__ = "0.1.0"
__author__ = "ZMLiang <lzzzmmgpt@gmail.com>"

from .main import main


# from cellniche import *
__all__ = ["main", "run", "Model", "Encoder"]
