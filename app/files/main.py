#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:30:19 2019

@author: scantleb
@brief: Generates types files, caffe learning settings and other file
structure admin, before making slurm files which are then submitted
to do GNINA jobs.

For training folds, keywords kinases, gpcrs, proteases, nucleases,
others and all (case sensitive) will cause trainining on all dude
targets of the specified class.
"""

# MASTER SCRIPT: GNINA TRAINING AND SCORING

import argparse
import os
import glob
import sys
import pathlib
import datetime
import numpy as np


kinases = ['abl1', 'braf', 'cdk2', 'fak1', 'igf1r', 'kpcb', 'mk01', 'mk10',
              'src', 'akt1', 'akt2', 'egfr', 'jak2', 'lck', 'met', 'tgfr1',
              'wee1', 'csf1r', 'kit', 'mapk2', 'mk14', 'mp2k1', 'plk1',
              'rock1', 'vgfr2']

proteases = ['ada17', 'casp3', 'dpp4', 'tryb1', 'urok',
        'ace', 'bace1', 'fa7', 'hivpr', 'fa10', 'thrb',
        'mmp13', 'lkha4', 'reni', 'try1']

nucleases = ['andr', 'ppard',
        'esr1', 'esr2', 'pparg', 'thb',
        'gcr', 'mcr', 'ppara', 'prgr', 'rxra']

gpcrs = ['drd3',
        'cxcr4',
        'aa2ar', 'adrb1', 'adrb2']

others = ['tysy', 'hdac8', 'hivrt', 'pur2', 'aofb', 'inha', 'comt', 'sahh',
        'pygm', 'fabp4', 'aldr', 'fnta', 'pa2ga', 'xiap', 'hmdh', 'dyr',
      'aces', 'pyrd', 'pgh1', 'parp1', 'cp2c9', 'def', 'pnph', 'pgh2', 'ada',
      'cp3a4', 'nram', 'fkb1a', 'ptn1', 'hdac2', 'gria2', 'glcm', 'cah2',
      'grik1', 'ital', 'dhi1', 'fpps', 'pde5a', 'nos1', 'kif11', 'hivint',
      'hxk4', 'kith', 'ampc', 'hs90a']

all_dude = kinases + proteases + gpcrs + nucleases + others

family_dict = {
        'kinases' : kinases,
        'proteases' : proteases,
        'others' : others,
        'gpcrs' : gpcrs,
        'nucleases' : nucleases,
        'all' : all_dude
        }

def readable_list(l):
    if type(l) == str and len(l) == 0:
        return 'Unspecified'
    if type(l) == dict:
        result = '{ '
        for key, value in l.items():
            if type(key) == str:
                result += "'" + key + "' : "
            else:
                result += str(key) + ' : '
            if type(value) == str:
                result += "' " + value + "', "
            elif type(value) == list:
                result += readable_list(value) + ', '
            else:
                result += str(value) + ', '
        return result[:-2] + ' }'
    elif type(l) == list:        
        result = '['
        for item in l:
            if type(item) == list or type(item) == dict:
                result += readable_list(item) + ', '
            elif type(item) != str:
                result += str(item) + ", "
            else:
                result += "'" + str(item) + "', "
        return result[:-2] + ']'
    return l

def get_script_path():
    return os.path.realpath(sys.path[0])

def longest_key(d):
    result = 0
    for key, value in d.items():
        if len(key) > result:
            result = len(key)
    return result

def generate_types_file(pdbids, params, gninatypes_basepath=
                        'chembl_test_data',
                        actives_list_base='resources/actives-lists',
                        decoys_list_base='resources/decoys-lists',
                        test=False):
    result = ''
    empty_gt = 'resources/empty_gninatype.gninatypes'
    if type(pdbids) == str:
        res = ''
        with open(pdbids, 'r') as f:
            for line in f.readlines():
                rec = empty_gt if (params['empty_receptor_test'] and test)\
                        or (params['empty_receptor_train'] and not test) \
                        else line.split()[1]
                res += '{0} {1} {2}\n'.format(line.split()[0], rec,
                        line.split()[2])
        return res[:-1]

    for pdbid in pdbids:
        
        if not (params['empty_receptor_test'] and test) and \
                not (params['empty_receptor_train'] and not test) :
            receptor_gt = os.path.join(gninatypes_basepath, 'receptors',
                    '{}.gninatypes'.format(pdbid))
        else:
            receptor_gt = empty_gt

        actives_list = os.path.join(actives_list_base,
                                    'actives-{}.txt'.format(pdbid))
        decoys_list = os.path.join(decoys_list_base,
                                    'decoys-{}.txt'.format(pdbid))
        
        with open(actives_list, 'r') as f:
            actives_text = f.readlines()
            alen = len(actives_text)
        with open(decoys_list, 'r') as f:
            decoys_text = f.readlines()
            dlen = len(decoys_text)
            
        actives_indexes = np.array(range(alen))
        decoys_indexes = np.array(range(dlen))
        if params['test_is_train']:
            dtrain_size = int(dlen*0.8)
            atrain_size = int(alen*0.8)
            
            decoys_train = np.random.choice(decoys_indexes, dtrain_size,
                    replace=False)
            actives_train = np.random.choice(actives_indexes, atrain_size,
                    replace=False)
            decoys_test = np.setdiff1d(decoys_indexes, decoys_train)
            actives_test = np.setdiff1d(actives_indexes, actives_train)
            
            res_train = ''
            res_test = ''
            
            for i in actives_train:
                ligand_gt = os.path.join(gninatypes_basepath, 'ligands',
                        pdbid, '{}.gninatypes'.format(actives_text[i].strip()))
                res_train += '1 {0} {1}\n'.format(receptor_gt, ligand_gt)

            for i in decoys_train:
                ligand_gt = os.path.join(gninatypes_basepath, 'ligands',
                        pdbid, '{}.gninatypes'.format(
                            decoys_text[i].strip()))
                res_train += '0 {0} {1}\n'.format(receptor_gt, ligand_gt)
                
            for i in actives_test:
                ligand_gt = os.path.join(gninatypes_basepath, 'ligands',
                        pdbid, '{}.gninatypes'.format(
                            actives_text[i].strip()))
                res_test += '1 {0} {1}\n'.format(receptor_gt, ligand_gt)

            for i in decoys_test:
                ligand_gt = os.path.join(gninatypes_basepath, 'ligands',
                        pdbid, '{}.gninatypes'.format(decoys_text[i].strip()))
                res_test += '0 {0} {1}\n'.format(receptor_gt, ligand_gt)

            return res_train[:-1], res_test[:-1]
        else:
            
            if not params['singly_redocked_test_set']:
                with open(actives_list, 'r') as f:
                    for line in f.readlines():
                        ligand_gt = os.path.join(gninatypes_basepath, 'ligands',
                                pdbid, '{}.gninatypes'.format(line.strip()))
                        if np.random.rand() < params['empty_probability'] and \
                                not test and params['empty_probability']:
                            result += '0 {0} {1}\n'.format(empty_gt, ligand_gt)
                        result += '1 {0} {1}\n'.format(receptor_gt, ligand_gt)
                
            if not (test and (params['redocked_actives_test_set'] or params['randomly_translated_actives_test_set'])):
                with open(decoys_list, 'r') as f:
                    for line in f.readlines():
                        ligand_gt = os.path.join(gninatypes_basepath, 'ligands',
                                pdbid, '{}.gninatypes'.format(line.strip()))
                        result += '0 {0} {1}\n'.format(receptor_gt, ligand_gt)
   

            if params['redocked_actives_test_set'] and test:
                for fname in glob.iglob('/homes/scantleb/redocked_actives_refined/gninatypes/{}/*decoy*.gninatypes'.format(pdbid)):
                    if int(fname.split('_')[-1].split('.')[0]) < 10:
                        result += '{2} {0} {1}\n'.format(receptor_gt, fname, 0)

            if params['randomly_translated_actives_test_set'] and test:
                for fname in glob.iglob('/homes/scantleb/translated_actives/gninatypes/{}/*.gninatypes'.format(pdbid)):
                    if int(fname.split('_')[-2]) < 10:
                        result += '{2} {0} {1}\n'.format(receptor_gt, fname, 0)

            if params['true_active_poses'] and (not test or not params['original_test_set']):
                for fname in glob.iglob('/homes/scantleb/redocked_actives_refined/\
                        gninatypes/{}/*active*.gninatypes'.format(pdbid)):

                    if int(fname.split('_')[-1].split('.')[0]) < \
                            params['true_active_poses']:
                        result += '{2} {0} {1}\n'.format(receptor_gt, fname, 1)

            if params['false_active_poses'] and (not test or not params['original_test_set']):
                for fname in glob.iglob('/homes/scantleb/redocked_actives_refined/gninatypes/{}/*decoy*.gninatypes'.format(pdbid)):
                    if int(fname.split('_')[-1].split('.')[0]) < params['false_active_poses']:
                        result += '{2} {0} {1}\n'.format(receptor_gt, fname, 0)

#            if params['translated_actives'] and (not test or not params['original_test_set']):
            if params['translated_actives'] and not test:
                for fname in glob.iglob('/homes/scantleb/translated_actives/gninatypes/{}/*.gninatypes'.format(pdbid)):
                    if int(fname.split('_')[-2]) < params['translated_actives']:
                        result += '{2} {0} {1}\n'.format(receptor_gt, fname, 0)

            if params['mixed_test_set_translated'] and test:
                for fname in glob.iglob('/homes/scantleb/translated_actives/gninatypes/{}/*.gninatypes'.format(pdbid)):
                    if int(fname.split('_')[-1].split('.')[0]) < 3:
                        result += '{2} {0} {1}\n'.format(receptor_gt, fname, 0)

            if params['mixed_test_set_redocked'] and test:
                for fname in glob.iglob('/homes/scantleb/redocked_actives_refined/gninatypes/{}/*decoy*.gninatypes'.format(pdbid)):                
                    if int(fname.split('_')[-1].split('.')[0]) < 3:
                        result += '{2} {0} {1}\n'.format(receptor_gt, fname, 0)

            if params['singly_redocked_test_set'] and test:
                for fname in glob.iglob('/homes/scantleb/redocked_actives_refined/gninatypes/{}/*decoy*.gninatypes'.format(pdbid)):
                    if not int(fname.split('_')[-1].split('.')[0]):
                        result += '{2} {0} {1}\n'.format(receptor_gt, fname, 1)



    return result[:-1]

def sets_from_file(fname):
    train, test = [], []
    with open(fname, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            l = line.replace(',', ' ').replace('[', '').replace(']',
                    '').replace("'", '')
            if len(l) == 0:
                continue
            if l.upper().find('TRAIN') != -1:
                for pdbid in l.split()[1:]:
                    train.append(pdbid.strip())
            elif l.upper().find('TEST') != -1:
                for pdbid in l.split()[1:]:
                    test.append(pdbid.strip())
            else:
                if len(train) == 0:
                    for pdbid in l.split():
                        train.append(pdbid.strip())
                elif len(test) == 0:
                    for pdbid in l.split():
                        test.append(pdbid.strip())
    return train, test

def parse_args(argv=None):

    parser = argparse.ArgumentParser(description=
            'Generate and execute GNINA experiemnt')
    
    parser.add_argument('-f', '--folds', type=str, required=True,
                        help='File containing lists of test and training \
                                PDBIDs')
    parser.add_argument('-b', '--basepath', type=str, required=True,
                        help='Path to set of related experiments')
    parser.add_argument('-n', '--experiment_name', type=str, required=True,
                        help='Name of specific experiment')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Prototxt file containing network architecture')
    parser.add_argument('-i', '--iterations', type=int, required=False,
                        help='Number of iterations for training')
    parser.add_argument('-c', '--empty_chance', type=float, required=False,
            default=0.0,
            help='Probability of using empty receptor for active (labelled 0)')
    parser.add_argument('-t', '--test_interval', type=int, required=False,
            default=1000000, help='Check test accuracy every t iterations')
    parser.add_argument('-s', '--batch_size', type=int, required=True,
            default=10, help='Modifies prototxt file to change batch size')
    parser.add_argument('--empty_receptor_train', action='store_true',
                        help='Train on all-zero gninatype file for receptor')
    parser.add_argument('--empty_receptor_test', action='store_true',
                        help='Test onall-zero gninatype file for receptor')
    parser.add_argument('-o', '--other_test_set', type=str, required=False,
            default='', help='Other test set to run trained model on at the \
                    end (*.types file)')
    parser.add_argument('-w', '--weights', type=str, required=False,
            default='', help='Bootstrap weights using caffemodel')
    parser.add_argument('--base_lr', type=float, required=False, default=-1,
            help='Base learning rate')
    parser.add_argument('-p', '--prefix', type=str, required=False, default='',
            help='Types prefix')
    parser.add_argument('--false_active_poses', type=int, default=0,
            help='Maximum number of extra high RMSD active poses (presumed to \
                    be good docks)')
    parser.add_argument('--true_active_poses', type=int, default=0,
            help='Maximum number of extra low RMSD active poses (presumed to \
                    be bad docks)')
    parser.add_argument('--translated_actives', type=int, default=0,
            help='Train on randonly translated/conformed actives (labelled 0)\
                    and number')
    parser.add_argument('--predict_only', action='store_true',
            help='Just predict')
    parser.add_argument('--original_test_set', action='store_true',
            help='Test set unmodified DUD-E folds')
    parser.add_argument('--redocked_actives_test_set', action='store_true',
            help='Test set consists 10 redocked active (labelled 0) and 1 true active\
                    (labelled 1)')
    parser.add_argument('--randomly_translated_actives_test_set', action='store_true',
            help='Test set consists 10 randomly translated/conformed actives (labelled 0)\
                    and 1 true active (labelled 1)')
    parser.add_argument('--mixed_test_set_translated', action='store_true',
            help='Test set consists normal DUD-E test folds, with added translated\
                    active decoys (labelled 0)')
    parser.add_argument('--mixed_test_set_redocked', action='store_true',
            help='Test set consists normal DUD-E test folds, with added redocked\
                    active decoys (labelled 0)')
    parser.add_argument('--singly_redocked_test_set', action='store_true',
            help='Test set consists normald DUD-E test folds, with actives substituted\
                    with single high-RMSD redocks')
    parser.add_argument('--train_only', action='store_true',
            help='Do not run predict.py on test set')
    args = parser.parse_args(argv)
    return args

def init(params, output_info):
    padding_len = max(longest_key(params), longest_key(output_info)) + 3
    
    experiment_root = os.path.join(params['basepath'], params['experiment_name'])
    caffe_output_path = os.path.join(experiment_root, 'caffe_output')
    params['ligmap'] = os.path.join(experiment_root, 'ligmap.old')
    
    
    pathlib.Path(experiment_root).mkdir(parents=True, exist_ok=True)
    pathlib.Path(caffe_output_path).mkdir(parents=True, exist_ok=True)
    os.system('cp resources/ligmap.old {}'.\
            format(experiment_root))
    
    train_set = os.path.join(experiment_root,
                             '{0}train0.types'.format(params['experiment_name']))
    test_set = os.path.join(experiment_root,
                            '{0}test0.types'.format(params['experiment_name']))

    # blank receptor
    test_set_br = os.path.join(experiment_root, '{0}test_noreceptor0.types'.\
            format(params['experiment_name']))

    if len(params['other_test_set']) > 0:
        other_test_set = os.path.abspath(params['other_test_set'])
    else:
        other_test_set = None

    return train_set, test_set, test_set_br, caffe_output_path, other_test_set

def change_batchsize(fname, batch_size):
    output = ''
    modify = False
    with open(fname, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if line.find('phase:') == -1 and line.find('batch_size') == -1:
                output += line
                continue
            if line.find('phase:') != -1:
                if line.find('TRAIN') != -1:
                    modify = True
                else:
                    modify = False
                output += line
                continue
            if line.find('batch_size') != -1 and modify:
                    tlen = len(line.split()[-1])
                    output += line[:-tlen-1] + str(batch_size) + '\n' 
                    modify = False
                    continue
            output += line
    return output
    
    
if __name__ == '__main__':

    kinases = ['abl1', 'braf', 'cdk2', 'fak1', 'igf1r', 'kpcb', 'mk01', 'mk10',
            'src', 'akt1', 'akt2', 'egfr', 'jak2', 'lck', 'met', 'tgfr1',
            'wee1', 'csf1r', 'kit', 'mapk2', 'mk14', 'mp2k1', 'plk1', 'rock1',
            'vgfr2']

    proteases = ['ada17', 'casp3', 'dpp4', 'tryb1', 'urok', 'ace', 'bace1',
            'fa7', 'hivpr', 'fa10', 'thrb', 'mmp13', 'lkha4', 'reni', 'try1']

    nucleases = ['andr', 'ppard', 'esr1', 'esr2', 'pparg', 'thb', 'gcr', 'mcr',
            'ppara', 'prgr', 'rxra']

    gpcrs = ['drd3', 'cxcr4', 'aa2ar', 'adrb1', 'adrb2']

    others = ['tysy', 'hdac8', 'hivrt', 'pur2', 'aofb', 'inha', 'comt', 'sahh',
            'pygm', 'fabp4', 'aldr', 'fnta', 'pa2ga', 'xiap', 'hmdh', 'dyr',
            'aces', 'pyrd', 'pgh1', 'parp1', 'cp2c9', 'def', 'pnph', 'pgh2',
            'ada', 'cp3a4', 'nram', 'fkb1a', 'ptn1', 'hdac2', 'gria2', 'glcm',
            'cah2', 'grik1', 'ital', 'dhi1', 'fpps', 'pde5a', 'nos1', 'kif11',
            'hivint', 'hxk4', 'kith', 'ampc', 'hs90a']

    all_dude = kinases + proteases + gpcrs + nucleases + others

    family_dict = {
        'kinases' : kinases,
        'proteases' : proteases,
        'others' : others,
        'gpcrs' : gpcrs,
        'nucleases' : nucleases,
        'all' : all_dude
        }
    
    args = parse_args()
    train, test = sets_from_file(args.folds)
    name = args.experiment_name
    args.basepath = os.path.abspath(args.basepath)
    
    params = {
        'basepath' : args.basepath,
        'experiment_name' : name,
        'train' : train,
        'test' : test,
        'predict_only': args.predict_only,
        'train_only': args.train_only,
        'layersfile' : args.model,
        'iterations' : args.iterations,
        'test_interval' : args.test_interval,
        'batch_size' : args.batch_size,
        'empty_probability' : args.empty_chance,
        'empty_receptor_train' : args.empty_receptor_train,
        'empty_receptor_test' : args.empty_receptor_test,
        'other_test_set': args.other_test_set,
        'true_active_poses' : args.true_active_poses,
        'false_active_poses' : args.false_active_poses,
        'translated_actives': args.translated_actives,
        'seed': np.random.randint(1000000),
        'test_is_train' : False,
        'original_test_set' : args.original_test_set,
        'redocked_actives_test_set' : args.redocked_actives_test_set,
        'randomly_translated_actives_test_set' : args.randomly_translated_actives_test_set,
        'mixed_test_set_translated' : args.mixed_test_set_translated,
        'mixed_test_set_redocked' : args.mixed_test_set_redocked,
        'singly_redocked_test_set' : args.singly_redocked_test_set
        }

    if params['original_test_set'] + params['redocked_actives_test_set'] + params['randomly_translated_actives_test_set']\
            + params['singly_redocked_test_set'] + params['mixed_test_set_redocked'] > 1:
        raise RuntimeError('Conflicting test set options!')

    output_info = {
            'weightsfile' : '{0}.0_iter_{1}.caffemodel'.format(name,
                args.iterations),
            'finaltest' : '{}.0.auc.finaltest'.format(name),
            'finaltrain' : '{}.0.auc.finaltrain'.format(name),
            'solverstate' : '{0}.0_iter_{1}.solverstate'.format(name,
                args.iterations),
            'logfile' : '{}.0.out'.format(name)
            }

    if len(args.weights) > 0:
        params['bootstrap_weights'] = args.weights

    if args.base_lr > 0:
        params['base_lr'] = args.base_lr

    train_set, test_set, test_set_noreceptor, caffe_output_path, \
            other_test_set = init(params, output_info)
    
    print('Initialisation complete.')
    test_done = False
    train_done = False
    real_test_set = os.path.join(params['basepath'], params['experiment_name'],
            'test_set.types')
    predictions_output = os.path.join(params['basepath'], params['experiment_name'],
                'caffe_output', 'predictions.txt')
    if params['original_test_set']:
        predictions_output = os.path.join(params['basepath'], params['experiment_name'],
                'caffe_output', 'predictions_original.txt')
    if params['redocked_actives_test_set']:
        predictions_output = os.path.join(params['basepath'], params['experiment_name'],
                'caffe_output', 'predictions_redocked.txt')
    if params['randomly_translated_actives_test_set']:
        predictions_output = os.path.join(params['basepath'], params['experiment_name'],
                'caffe_output', 'predictions_translated.txt')
    if params['mixed_test_set_translated']:
        predictions_output = os.path.join(params['basepath'], params['experiment_name'],
                'caffe_output', 'predictions_mixed_translated.txt')
    if params['mixed_test_set_redocked']:
        predictions_output = os.path.join(params['basepath'], params['experiment_name'],
                'caffe_output', 'predictions_mixed_redocked.txt')
    if params['singly_redocked_test_set']:
        predictions_output = os.path.join(params['basepath'], params['experiment_name'],
                'caffe_output', 'predictions_singly_redocked.txt')        
    if params['empty_receptor_test']:
        predictions_output = os.path.join(params['basepath'], params['experiment_name'],
                'caffe_output', 'predictions_no_receptor.txt')

    x = 1
    while os.path.isfile(predictions_output):
        predictions_output = '{0}_{1}.{2}'.format(predictions_output.split('.')[0], x,
                predictions_output.split('.')[-1])
        x += 1


    x=1
    while os.path.isfile(real_test_set):
        real_test_set = os.path.join(params['basepath'], params['experiment_name'],
            'test_set_{}.types'.format(x))
        x += 1

    params['test_set_types_file'] = real_test_set
    params['predictions_output_file'] = predictions_output

    # Pregenerated train types files
    if params['train'][0].find('.types') != -1 and not args.predict_only:
        cp_cmd = 'cp {0} {1}'.format(os.path.abspath(params['train'][0]),
                train_set)
        print('Pregenerated train types file; copying to base directory:\n',
                cp_cmd)
        os.system(cp_cmd)
        params['train'] = train_set
        train_done = True

    # Pregenerated test types files
    if params['test'][0].find('.types') != -1 and not params['train_only']:
        cp_cmd = 'cp {0} {1}'.format(os.path.abspath(params['test'][0]),
                real_test_set)
        print('Pregenerated test types file; copying to base directory:\n',
                cp_cmd)
        os.system(cp_cmd)
        params['test'] = real_test_set
        test_done = True

    # Protein target family keyword detection: train on whole DUD-E for that
    # family
    if not args.predict_only:
        if params['train'][0] == 'kinases' or \
                params['train'][0] == 'others' or \
                params['train'][0] == 'nucleases' or \
                params['train'][0] == 'gpcrs' or \
                params['train'][0] == 'proteases' or \
                params['train'][0] == 'all':
            traintext = generate_types_file(family_dict[params['train'][0]],
                                            test=False,
                                            params=params)
            with open(train_set, 'w') as f:
                f.write(traintext)
            train_done = True

    # Training/test sets comprise DUD-E targets specified by list in folds
    # text file
    if not train_done and not args.predict_only:
        traintext = generate_types_file(params['train'],
                                        test=False,
                                        params=params)
        with open(train_set, 'w') as f:
            f.write(traintext)
    if not test_done and not params['train_only']:
        testtext = generate_types_file(params['test'],
                                       test=True, params=params)
        with open(real_test_set, 'w') as f:
            f.write(testtext)

    # If test and train targets are same, select random 80:20 split for
    # training and testing. NOT RECOMMENDED.
    params['test_is_train'] = False
    if params['test'] == params['train']:
        params['test_is_train'] = True
        traintext, testtext = generate_types_file(params['train'], params=params)

    # Modify caffe prototxt for specified training batch size

    if not args.predict_only:
        prototxt = change_batchsize(params['layersfile'], params['batch_size'])
        with open(params['layersfile'], 'w') as f:
            f.write(prototxt)
    
    print('Types files generated.')
    
    experiment_root = os.path.join(params['basepath'], params['experiment_name'])
    params_md_path = os.path.join(experiment_root, 'PARAMS.MD')
    if params['predict_only']:
        params_md_path = os.path.join(experiment_root, 'PARAMS_PREDICT_0.MD')
        x = 1
        while os.path.isfile(params_md_path):
            params_md_path = os.path.join(experiment_root, 'PARAMS_PREDICT_{}.MD'.format(x))
            x += 1

    padding_len = max(longest_key(params), longest_key(output_info)) + 3

    params_str = 'Time of experiment generation: {:%d-%m-%Y %H:%M:%S}\n\n'.\
            format(datetime.datetime.now())

    params_str += '------PARAMETERS------\n'
    for key, value in params.items():
        params_str += '{0}{1}{2}\n'.format(key, ' '*(padding_len-len(key)),
                       readable_list(value))
    params_str += '\n-------OUTPUTS--------\n'
    for key, value in output_info.items():
        params_str += '{0}{1}{2}\n'.format(key, ' '*(padding_len-len(key)),
                       readable_list(value))

    print(params_str)

    with open(params_md_path, 'w') as f:
        f.write(params_str)


    # We use a dummy types file for the automatic end of training test run
    # because we will use predict.py to test on our real test set. This gives
    # output labelled by gninatypes file rather than 'anonymous'

    if not args.predict_only:
        os.chdir(os.path.join(params['basepath'], name))
        cp_cmd = 'cp ~/gnina_scripts/resources/empty.types {0}/{2}/{1}test0.types'\
                .format(params['basepath'], params['experiment_name'],
                        params['experiment_name'])
        os.system(cp_cmd)

        # Everything is in place now, train the network
        traincmd = 'python3 scripts/train.py -m {0} -p {1} -i {2} -t {3} -o {4} -s {5}'
        traincmd = traincmd.format(
                params['layersfile'],
                params['experiment_name'],
                params['iterations'],
                params['test_interval'],
                params['experiment_name'],
                params['seed'])
        try:
            params['bootstrap_weights']
        except Exception:
            pass
        else:
            # Starting weights specified (probably finetuning). Use these.
            traincmd += ' --weights {}'.format(params['bootstrap_weights'])

        try:
            params['base_lr']
        except Exception:
            pass
        else:
            traincmd += ' --base_lr {}'.format(params['base_lr'])

        print(traincmd)
        os.system(traincmd)
    
        print('Training complete.')

        # GNINA spits out files in an annoying place. Move them to
        # caffe_output
        mvcmd = 'mv {0}/*.0* {1}'.format(os.path.join(params['basepath'],
            params['experiment_name']), caffe_output_path)
        os.system(mvcmd)
        print('Files moved.')


    if not params['train_only']:
 
        # GNINA only tested on our empty dummy set; now we use the caffemodel to
        # train on our real test set.
        os.chdir(params['basepath'])
        os.system('cp resources/ligmap.old {}'.format(
            params['basepath']))
        predcmd = 'python3 scripts/predict.py -m {0} \
                -w {1} -i {2} >> {3}'
        cm_cmd = os.path.join(caffe_output_path, '*.caffemodel')
        print('Looking for caffemodel using glob command:', cm_cmd)
        try:
            caffemodel = next(glob.iglob(os.path.join(caffe_output_path, '*.caffemodel')))
        except StopIteration:
            os.system('cp {0} {1}'.format(params['bootstrap_weights'], caffe_output_path))
            caffemodel = next(glob.iglob(os.path.join(caffe_output_path, '*.caffemodel')))

        test_output = os.path.join(caffe_output_path, 'test_predictions.txt')
        x=1
        with open(params['predictions_output_file'], 'w') as f:
            f.write(params_str)
        predcmd_1 = predcmd.format(
                params['layersfile'],
                caffemodel,
                real_test_set,
                params['predictions_output_file']
                )
        print('Predcmd:', predcmd_1)
        os.system(predcmd_1)
    print('Finished! Exiting...')



















