# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:28:50 2019

@author: Ashima
"""

import torch
print(torch.rand(2, 3).cuda())
print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

