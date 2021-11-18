import numpy as np
import pypcd
import map_evaluation

pcd_file = "/media/hyx/49ba7b04-2195-48ba-b695-865c260fb515/garage.pcd"

# map_evaluation.init(pcd_file=pcd_file)

pc = np.fromfile("/media/hyx/49ba7b04-2195-48ba-b695-865c260fb515/label_pc/sequences/01/velodyne/000000.bin", dtype=np.float32).reshape((-1, 4))
map_evaluation.init(input=pc)

print(map_evaluation.mme())
print(map_evaluation.mpv())
