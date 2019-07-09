import torch
import torchvision.utils as vutils
import numpy as np
import sys
import os
import cv2
np.set_printoptions(precision=3, suppress=True, threshold=sys.maxsize)
import pdb

C = torch.ones(1, 1, 40, 12)
counter = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
counter.weight.data[...] = 1.0
neighbor_count = counter(C).data
potential_thres = 0.5 * neighbor_count
Ms = torch.zeros(4, 1, 1, 40, 12)
new_prof_dir = 'D:\\datagen\\conv_dataset\\data12f2'
if not os.path.isdir(new_prof_dir):
    os.mkdir(new_prof_dir)
    
written_file_num = 0
for i in range(8000, 20000):
    Ms[0] = torch.randint(2, size=(1, 1, 40, 12))
    prof_filename = "%d.png" %i
     
    for j in range(3):
        M_potential = counter(Ms[j])
        Ms[j+1][M_potential <  potential_thres] = 0
        Ms[j+1][M_potential >  potential_thres] = 1
        # avoid the slight bias towards 1
        Ms[j+1][M_potential == potential_thres] = Ms[j][M_potential == potential_thres]
    
    new_prof_filepath = os.path.join(new_prof_dir, prof_filename)
    new_prof_img = Ms[3, 0, 0].data.numpy() * 255.0
    new_prof_img = new_prof_img.astype(np.uint8)
    cv2.imwrite(new_prof_filepath, new_prof_img)
    written_file_num += 1
    
print("%d files generated." %written_file_num)
