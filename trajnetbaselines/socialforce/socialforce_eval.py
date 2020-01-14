"""Command line tool to create a table of evaluations metrics."""
import argparse
from itertools import compress

import pickle
import numpy as np

import trajnettools
# import trajnetbaselines
from trajnettools import show
from . import socialforce 
from . import orca 
from trajnetbaselines import kalman 
from trajnettools.interactions import collision_avoidance

class Evaluator(object):
    def __init__(self, scenes, dest_dict=None, params=None):
        self.scenes = scenes
        self.dest = dest_dict
        self.params = params

        self.average_l2 = {'N': len(scenes)}
        self.final_l2 = {'N': len(scenes)}

    def aggregate(self, name, predictor, dest_type='true'):
        print('evaluating', name)

        average = 0.0
        final = 0.0

        pred_dict = {}
        pred_neigh_dict = {}
        n = 0

        for scene_i, paths in enumerate(self.scenes):
            if 'sf' in name or 'orca' in name:
                ##filter for interactions
                rows = trajnettools.Reader.paths_to_xy(paths)
                neigh_paths = paths[1:]
                interaction_index = collision_avoidance(rows)
                neigh = list(compress(neigh_paths, interaction_index))
                paths = [paths[0]] + neigh

                prediction, neigh = predictor(paths, self.dest, dest_type, self.params,  predict_all=True)[0]

                ## visualize
                pred_dict['orca'] = prediction
                pred_neigh_dict['orca'] = neigh 

                n += 1
                if n < 17:
                    with show.predicted_paths(paths, pred_dict, pred_neigh_paths=pred_neigh_dict):
                        pass
                else:
                    break

            else:
                prediction, neigh = predictor(paths)[0]

            average_l2 = trajnettools.metrics.average_l2(paths[0], prediction)
            final_l2 = trajnettools.metrics.final_l2(paths[0], prediction)

            # aggregate
            average += average_l2
            final += final_l2

        average /= len(self.scenes)
        final /= len(self.scenes)

        self.average_l2[name] = average
        self.final_l2[name] = final

        return self

    def result(self):
        return self.average_l2, self.final_l2


def eval(input_file, dest_file, simulator, params, type_ids=None):
    print('dataset', input_file)

    reader = trajnettools.Reader(input_file, scene_type='paths')
    scenes = [s for _, s in reader.scenes()]

    ## Type of scene
    if type_ids is None:
        trajectory_type = 3
        interaction_type = 2
        type_ids = [scene_id for scene_id in reader.scenes_by_id \
                    if interaction_type in reader.scenes_by_id[scene_id].tag[1]]

    print(type_ids)
    scenes = [scenes[type_id] for type_id in type_ids]

    dest_dict = None
    if dest_file is not None:
        dest_dict = pickle.load(open(dest_file, "rb"))

    evaluator = Evaluator(scenes, dest_dict, params)

    # ORCA
    for dest_type in ['interp']:
        evaluator.aggregate(simulator + dest_type, orca.predict, dest_type)
    # Social Force
    # for dest_type in ['pred_end']:
    #     evaluator.aggregate('sf' + dest_type, socialforce.predict, dest_type)
    # evaluator.aggregate('kf', kalman.predict)

    return evaluator.result()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--simulator', default='sf',
                        choices=('orca', 'sf', 'kalman'))

    parser.add_argument('--tau', default=0.5, type=float,
                        help='Tau of Social Force')
    parser.add_argument('--vo', default=4.0, type=float,
                        help='V0 of Social Force')
    parser.add_argument('--sigma', default=0.3, type=float,
                        help='sigma of Social Force')

    parser.add_argument('--min_dist', default=4, type=float,
                        help='MinNeighDist of ORCA')
    parser.add_argument('--react_time', default=4, type=float,
                        help='NeighReactTime of ORCA')
    parser.add_argument('--radius', default=0.6, type=float,
                        help='agent radius of ORCA')

    args = parser.parse_args()

    if args.simulator == 'sf':
        params = [args.tau, args.vo, args.sigma]
    elif args.simulator == 'orca':
        params = [args.min_dist, args.react_time, args.radius]
    else:
        params = None

    datasets = [
        'DATA_BLOCK/data/train/real_data/biwi_hotel.ndjson',
        # 'DATA_BLOCK/data/train/real_data/crowds_zara01.ndjson',
        # 'DATA_BLOCK/data/train/real_data/crowds_zara03.ndjson',
        # 'DATA_BLOCK/data/train/real_data/crowds_students001.ndjson',
        # 'DATA_BLOCK/data/train/real_data/crowds_students003.ndjson',
        # 'DATA_BLOCK/data/train/real_data/lcas.ndjson',
        # 'DATA_BLOCK/data/train/real_data/wildtrack.ndjson',

        # 'DATA_BLOCK/data/groundtruth/real_data/biwi_eth.ndjson',
        # 'DATA_BLOCK/data/groundtruth/real_data/crowds_zara02.ndjson',
        # 'DATA_BLOCK/data/groundtruth/real_data/crowds_uni_examples.ndjson',
    ]
    # base = 'dest_new'
    dest_dicts = [
        'dest_new/biwi_hotel.pkl',
        # 'dest_new/crowds_zara01.pkl',
        # 'dest_new/crowds_zara03.pkl',
        # 'dest_new/crowds_students001.pkl',
        # 'dest_new/crowds_students003.pkl',
        # 'dest_new/lcas.pkl',
        # 'dest_new/wildtrack.pkl',

        # 'dest/biwi_eth.pkl',
        # 'dest/crowds_zara02.pkl',
        # 'dest/crowds_uni_examples.pkl',        
    ]
    
    ## Interaction Scenes
    filtered_ids = {}
    filtered_ids['biwi_hotel'] = [92, 116, 118, 121, 123, 174, 233] # 102, 103, 175, 176
    filtered_ids['crowds_zara01'] = [13, 107, 108, 115, 117, 119, 120, 225, 226, 270, 320, 321, \
                                     339, 345, 367, 390, 394, 395, 396, 397, 411, 586, 587, 748, \
                                     750, 751, 789, 806, 807, 824, 837, 838, 839, 840, 929, 931]
    filtered_ids['crowds_zara03'] = [46, 47, 79, 80, 97, 200, 202, 256, 257, 258, 342, 343, 368, \
                                     449, 447, 450, 590, 703, 704, 755, 772, 855, 856, 877, 893]

    # filtered_ids['biwi_hotel'] = [233] # 103, 116
    filtered_ids['crowds_zara01'] = [824, 930]
    filtered_ids['crowds_zara03'] = [772]

    results = {}
    for i, dataset in enumerate(datasets):
        dataset_name = dataset.replace('DATA_BLOCK/data/train/real_data/', '').replace('.ndjson', '')
        type_ids = filtered_ids[dataset_name]        
        results[dataset_name] = eval(dataset, dest_dicts[i], args.simulator, params, type_ids)

    # results = {dataset
    #            .replace('DATA_BLOCK/data/groundtruth/real_data/', '')
    #            .replace('.ndjson', ''): eval(dataset, dest_dicts[i])
    #            for i, dataset in enumerate(datasets)}

    exit()

    with open("orca.txt", "a") as myfile:
        myfile.write('params: {}, {}, {} \n'.format(*params))
        print('params', *params)

        myfile.write('## Average L2 [m]\n')
        print('## Average L2 [m]')

        myfile.write('{dataset:>30s} |   N  | True | Int \n'.format(dataset=''))
        print('{dataset:>30s} |   N  | True | Int '.format(dataset=''))

        for dataset, (r, _) in results.items():
            print(
                '{dataset:>30s}'
                ' | {r[N]:>4}'
                ' | {r[orcatrue]:.2f}'
                ' | {r[orcainterp]:.2f}'.format(dataset=dataset, r=r)
            )
            myfile.write(
                        '{dataset:>30s}'
                        ' | {r[N]:>4}'
                        ' | {r[orcatrue]:.2f}'
                        ' | {r[orcainterp]:.2f} \n'.format(dataset=dataset, r=r)
            ) 

        myfile.write('\n')    
        print('')

        myfile.write('## Final L2 [m] \n')
        print('## Final L2 [m]')

        myfile.write('{dataset:>30s} |   N  | True | Int \n'.format(dataset=''))
        print('{dataset:>30s} |   N  | True | Int '.format(dataset=''))
        
        for dataset, (_, r) in results.items():
            print(
                '{dataset:>30s}'
                ' | {r[N]:>4}'
                ' | {r[orcatrue]:.2f}'
                ' | {r[orcainterp]:.2f}'.format(dataset=dataset, r=r)
            )
            myfile.write(
                        '{dataset:>30s}'
                        ' | {r[N]:>4}'
                        ' | {r[orcatrue]:.2f}'
                        ' | {r[orcainterp]:.2f} \n \n \n '.format(dataset=dataset, r=r)
            )             

    # print('## Average L2 [m]')
    # print('{dataset:>30s} |   N  | True | Int | Vel | KF'.format(dataset=''))
    # for dataset, (r, _) in results.items():
    #     print(
    #         '{dataset:>30s}'
    #         ' | {r[N]:>4}'
    #         ' | {r[sftrue]:.2f}'
    #         ' | {r[sfinterp]:.2f}'
    #         ' | {r[sfvel]:.2f}'
    #         ' |  {r[kf]:.2f}'.format(dataset=dataset, r=r)
    #     )
    #         # 
    # print('')
    # print('## Final L2 [m]')
    # print('{dataset:>30s} |   N  | True | Int | Vel | KF'.format(dataset=''))
    # for dataset, (_, r) in results.items():
    #     print(
    #         '{dataset:>30s}'
    #         ' | {r[N]:>4}'
    #         ' | {r[sftrue]:.2f}'
    #         ' | {r[sfinterp]:.2f}'
    #         ' | {r[sfvel]:.2f}'
    #         ' |  {r[kf]:.2f}'.format(dataset=dataset, r=r)
    #     )

if __name__ == '__main__':
    main()