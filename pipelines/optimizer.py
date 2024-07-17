import math
import threading
import time
#import cupy as cp
import numpy as np
from pygco import cut_from_graph
from timeit import default_timer as timer
import progressbar

class Optimizer(object):

    def __init__(self):
        pass

    # 'random'):
    def run(self, mesh, patch_prob_output, num_classes, X, round_factor=100, max_nm=5000, split_way='split'):
        print('Refining by pygco...')
        global cell_ids
        global normals
        global barycenters
        global g_edges
        global _ncells
        global cell_ids_p
        global cell_ids_c
        
        _ncells = mesh.NCells()
        patch_prob_output[patch_prob_output < 1.0e-6] = 1.0e-6

        # unaries
        print('Calculating unaries...')
        ppo = np.asarray(patch_prob_output)
        unaries = -round_factor * np.log10(ppo)
        unaries = unaries.astype(np.int32)
        unaries = unaries.reshape(-1, num_classes)

        # pairwise
        print('Calculating pairwise...')
        pairwise = (1 - np.eye(num_classes, dtype=np.int32))
      
        print('calculating mesh_normals')

        # edges
        normals = X[-1][-3:].transpose(1, 0).cpu().detach().numpy()
        #cells = original_cells_d.copy()
        barycenters = np.asarray(mesh.cellCenters())  # don't need to copy        
        cell_ids = np.asarray(mesh.faces())

        
        #cell_ids = np.sort(cell_ids, axis=1)

        print('calculating auxiliary cell_ids array')
        cell_ids_p = [np.array([]) for i in range(mesh.points().shape[0])]
        cell_ids_c = [np.array([]) for i in range(mesh.points().shape[0])]

        progress = progressbar.ProgressBar()
        
        for i in progress(range(_ncells)):
            item = cell_ids[i]
            for x in item:
                #cell_ids_p[x].append(item)
                if cell_ids_p[x].shape[0] == 0:
                    cell_ids_p[x] = np.array([item])
                    cell_ids_c[x] = np.array([i])
                else:
                    cell_ids_p[x] = np.unique(np.concatenate([cell_ids_p[x], [item]]), axis=0)
                    cell_ids_c[x] = np.unique(np.concatenate([cell_ids_c[x], [i]]), axis=0)
        cell_ids_p = np.array(cell_ids_p)
        cell_ids_c = np.array(cell_ids_c)
        #return

        lambda_c = 30

        index = 0
        step = 10000
        g_edges=[np.empty([1, 3], order='C') for _ in range(math.ceil(_ncells/step))] 
        time_start = timer()
        threads = list()
        
        for i_node in range(0, _ncells, step):
            print(f"Starting thread: {index}")
            last = min(i_node + step, _ncells)
            #calc_neighbours(i_node, last, index)
            x = threading.Thread(target=calc_neighbours, args=(i_node, last, index,))
            threads.append(x)
            x.start()
            index +=1

        for index, thread in enumerate(threads):
            thread.join()
        
        time_end = timer()

        print(f"Elapsed time in threads: {time_end - time_start} seconds")
        
        edges = np.concatenate(g_edges)

        edges[:, 2] *= lambda_c*round_factor
        edges = edges.astype(np.int32)
        print(edges.shape)

        #return
        print('calling gcuts')
        start = time.perf_counter()
        refine_labels = cut_from_graph(edges, unaries, pairwise, 10)
        print(refine_labels.shape)
        end = time.perf_counter()
        print(f"gCuts Elapsed time: {end-start} seconds")

        refine_labels = refine_labels.reshape([-1, 1])
        #mesh3 = mesh.clone()
        return mesh.clone(), refine_labels

def calc_neighbours(start, end, thread_index):

    local_edges = np.empty([1, 3], order='C')
    t_start = timer()
    for i_node in range(start, end):
        # Find neighbors
        nei = np.sum(np.isin(np.concatenate([cell_ids_p[i_] for i_ in cell_ids[i_node, :]]), cell_ids[i_node, :]), axis=1)
        places = np.concatenate([cell_ids_c[i_] for i_ in cell_ids[i_node, :]])
        nei_id = np.where(nei == 2)
        #if i_node % 1000 == 0:
        #    print(i_node, nei_id)
        for _i_nei in nei_id[0][:]:
            i_nei = places[_i_nei]
            if i_node < i_nei:
                cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])/np.linalg.norm(
                    normals[i_node, 0:3])/np.linalg.norm(normals[i_nei, 0:3])
                if cos_theta >= 1.0:
                    cos_theta = 0.9999
                theta = np.arccos(cos_theta)
                phi = np.linalg.norm(
                    barycenters[i_node, :] - barycenters[i_nei, :])
                if theta > np.pi/2.0:
                    local_edges = np.concatenate((local_edges, np.array(
                        [i_node, i_nei, -np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
                else:
                    beta = 1 + \
                        np.linalg.norm(
                            np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                    local_edges = np.concatenate((local_edges, np.array(
                        [i_node, i_nei, -beta*np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
    local_edges = np.delete(local_edges, 0, 0)
    g_edges[thread_index] = local_edges


