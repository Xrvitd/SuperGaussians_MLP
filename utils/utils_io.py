from datetime import datetime
import os, sys, logging, re
import shutil
import subprocess
from pathlib import Path
import glob #, GPUtil
import h5py, json
import numpy as np
from collections import OrderedDict

import torch

from tqdm import tqdm
from functools import partial
from multiprocessing import pool, Pool
from copy import deepcopy

import pickle


def read_data_binary(fileName, shape):
    """
    Reads a depth map from a binary file and reshapes it to the given shape.

    Args:
    - fileName (str): The path to the binary file containing the depth map.
    - shape (tuple): The shape of the depth map (height, width).

    Returns:
    - numpy.ndarray: The depth map as a 2D NumPy array of floats.
    """
    if not os.path.exists(fileName):
        return None
    with open(fileName, 'rb') as f:
        # Read the entire file into a bytes object
        data = f.read()

        # Convert the bytes object to a 1D numpy array of type float32
        depthMap = np.frombuffer(data, dtype=np.float32)

        # Check if the total number of floats matches the product of the dimensions
        if depthMap.size != shape:
            raise ValueError("The size of the file does not match the specified shape")

        # Reshape the 1D array to the specified shape (height, width)
        depthMap = depthMap.reshape(shape)

    return depthMap


def write_data_to_binary(fileName, depthMap):
    with open(fileName, 'wb') as f:
        # Assuming depthMap is a NumPy array of dtype=np.float32
        f.write(depthMap.tobytes())



def write_args_to_json(path_args, args):
    # write command line arguments to a file
    ensure_pdir_existence(path_args)
    with open(path_args, 'w') as f:
        args_dump = deepcopy(args)
        if 'device' in args_dump.__dict__:
            args_dump.device = None
        args_dict = vars(args_dump)
        json.dump(args_dict, f, indent=2)


#### new funcs above

def start_timer():
    return datetime.now()

def print_consumed_time(t_start, msg='Consumed time: '):
    t_end = datetime.now()
    print(f'{msg} {(t_end-t_start).total_seconds():.02f} sec')

def split_list_to_chunks(lst, n_chunks):
    chunk_size = len(lst)//n_chunks + 1
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def run_func_multithread(func, n_thread, lis_args):
    print(f'Process: {len(lis_args)} items, using thread: {n_thread}')
    t1 = datetime.now()
    if n_thread==1:
        lis_result = []
        for args in tqdm(lis_args):
            result = func(args)
            lis_result.append(result)
        return lis_result

    with pool.ThreadPool(n_thread) as p:
        # all_data = p.map(func, lis_args)
        all_data = list( tqdm(p.imap_unordered(func, lis_args), total=n_thread))

    print(f'Consumed time: {(datetime.now()-t1).total_seconds()/60:.02f} min')
    return all_data

def run_func_multi_process(func, n_process, lis_args):
    print(f'Process: {len(lis_args)} items, using process: {n_process}')
    t_start = start_timer()
    if n_process==1:
        lis_result = []
        for args in tqdm(lis_args):
            result = func(args)
            lis_result.append(result)
        return lis_result

    # with Pool(n_process) as p:
    #     all_data_chunks = p.map(func, lis_args)

    # with pool.ThreadPool(n_thread) as p:
    #     # all_data = p.map(func, lis_args)
    #     all_data = list( tqdm(p.imap_unordered(func, lis_args), total=n_thread))

    with Pool(n_process) as p:
        # all_data_chunks = p.map(func, lis_args)
        all_data_chunks = list( tqdm(p.imap_unordered(func, lis_args), total=n_process))

    print_consumed_time(t_start)
    return all_data_chunks

def run_func_multiprocess_multithread(func_thread, n_process, n_thread, lis_args):
    # func thread with arguments
    t_start = start_timer()
    if n_process==1:
        print(f'Models to process: {len(lis_args)}. Thread used: {1}')

        lis_result = []
        for args in tqdm(lis_args):
            result = func_thread(args)
            lis_result.append(result)

        print_consumed_time(t_start)
        return lis_result

    print(f'Models to process: {len(lis_args)}. Thread used: {n_process*n_thread}')

    lis_chunks = split_list_to_chunks(lis_args, n_chunks=n_process)
    with Pool(n_process) as p:
        all_data_chunks = p.map(partial(run_func_multithread, func_thread, n_thread), lis_chunks)
        # all_data_chunks = list( tqdm(p.imap_unordered(partial(run_func_multithread, func_thread, n_thread), lis_chunks) ))

    all_data_merge = []
    for chunk in all_data_chunks:
        all_data_merge += chunk
    print_consumed_time(t_start, msg='Total consumed time: ')
    return all_data_merge

def load_json_file(fn):
    with open(fn, 'r') as f:
        root_json = json.load(f)
    return root_json

# hdf5
def load_hdf5_data(path_file, data_id = -1):
    with h5py.File(path_file, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        keys = list(f.keys())
        data_all = {}
        for key in keys:
            # a_group_key = list(f.keys())[data_id]

            # Get the data
            # data_curr = list(f[a_group_key])
            data_all[key] = np.array(f[key])
    # print(data_all.shape)
    return data_all

# Path
def checkExistence(path):
    if not os.path.exists(path):
        return False
    else:
        return True

def check_existence(path):
    return checkExistence(path)

def ensureDirExistence(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            logging.info(f"Dir is already existent: {dir}")
    except Exception:
        logging.error(f"Fail to create dir: {dir}")
        # exit()

def ensure_dir_existence(dir):
    ensureDirExistence(dir)

def ensure_pdir_existence(path_file):
    path = Path(path_file)
    ppath = str(path.parent)
    ensure_dir_existence(ppath)

def get_path_components(path):
    path = Path(path)
    ppath = str(path.parent)
    stem = str(path.stem)
    ext = str(path.suffix)
    return ppath, stem, ext

def add_file_name_suffix(path_file, suffix):
    ppath, stem, ext = get_path_components(path_file)

    path_name_new = ppath + "/" + stem + str(suffix) + ext
    # print(f"File name with suffix: {path_name_new}")
    return path_name_new

def add_file_name_prefix(path_file, prefix, check_exist = True):
    '''Add prefix before file name
    '''
    ppath, stem, ext = get_path_components(path_file)
    path_name_new = ppath + "/" + str(prefix) + stem  + ext

    if check_exist:
        ensureDirExistence(ppath + "/" + str(prefix))

    # print(f"File name with prefix: {path_name_new}")
    return path_name_new

def add_file_name_prefix_and_suffix(path_file, prefix, suffix, check_exist = True):
    path_file_p = add_file_name_prefix(path_file, prefix, check_exist = True)
    path_file_p_s = add_file_name_suffix(path_file_p, suffix)
    return path_file_p_s

def get_files_stem(dir, ext_file):
    '''Get stems of all files in directory with target extension
    Return:
        vec_stem
    '''
    vec_path = sorted(glob.glob(f'{dir}/**{ext_file}'))
    vec_stem = []
    for i in range(len(vec_path)):
        pparent, stem, ext = get_path_components(vec_path[i])
        vec_stem.append(stem)
    return vec_stem

def get_files_path(dir, ext_file):
    return sorted(glob.glob(f'{dir}/**{ext_file}'))

# IO
def read_pickle_to_class_instance(path_pkl):
    if not check_existence(path_pkl):
        print(f'File not existent: {path_pkl.split("/")[-6:-1]}')
    with open(path_pkl, "rb") as infile:
        loaded_instance = pickle.load(infile)
    return loaded_instance

def save_class_instance_to_pickle(path_pkl, ins_save, ensure_pdir_exist=True):
    if ensure_pdir_exist:
        ensure_pdir_existence(path_pkl)

    with open(path_pkl, "wb") as outfile:
        pickle.dump(ins_save, outfile)

def read_txt_to_list(path_txt, mode_split=None, lis_idx_select=None, is_float=False):
    fTxt = open(path_txt, "r")
    lines = fTxt.readlines()
    lis = []
    for line in lines:
        line = line.split('\n')[0]
        if mode_split is not None:
            line = list(filter(None, re.split(mode_split, line)))
            if lis_idx_select is not None:
                line = [line[i] for i in lis_idx_select]

            if is_float:
                line = [float(item) for item in line]
        lis.append(line)
    return lis

def copy_file(source_path, target_dir):
    try:
        ppath, stem, ext = get_path_components(target_dir)
        ensure_dir_existence(ppath)
        if check_existence(target_dir):
            logging.info(f"File is already existent: {target_dir}")
            return
        shutil.copy(source_path, target_dir)
    except Exception:
        logging.error(f"Fail to copy file: {source_path}")
        # exit(-1)

def remove_dir(dir):
    try:
        shutil.rmtree(dir)
    except Exception as ERROR_MSG:
        logging.error(f"{ERROR_MSG}.\nFail to remove dir: {dir}")
        exit(-1)

def remove_file(path_file):
    try:
        os.remove(path_file)
    except Exception as ERROR_MSG:
        logging.error(f"{ERROR_MSG}.\nFail to remove dir: {dir}")
        exit(-1)

def copy_lis_dir(lis_dir, dir_target):
    for item in lis_dir:
        copy_dir(item, f"{dir_target}/{item}" )

def copy_dir(source_dir, target_dir):
    try:
        if not os.path.exists(source_dir):
            logging.error(f"source_dir {source_dir} is not exist. Fail to copy directory.")
            exit(-1)

        if not os.path.exists(target_dir):
            shutil.copytree(source_dir, target_dir)
        else:
            logging.error(f"Existed target_dir: {target_dir}. Skip Copy.")
        #     exit(-1)
    except Exception as ERROR_MSG:
        logging.error(f"{ERROR_MSG}.\nFail to copy file: {source_dir}")
        exit(-1)

def INFO_MSG(msg):
    print(msg)
    sys.stdout.flush()

def changeWorkingDir(working_dir):
    try:
        os.chdir(working_dir)
        print(f"Current working directory is { os.getcwd()}.")
    except OSError:
        print("Cann't change current working directory.")
        sys.stdout.flush()
        exit(-1)

def run_subprocess(process_args):
    # fLog = open(path_log, "a")
    pProcess = subprocess.Popen(process_args)
    #writeLogFile(pProcess, path_log)
    pProcess.wait()
    # fLog.close()


def ensureDirExistence(dir):
    try:
        if not os.path.exists(dir):
            # INFO_MSG(f"Create directory: {dir}.")
            os.makedirs(dir)
    except Exception:
        INFO_MSG(f"Fail to create dir: {dir}")
        # exit(-1)

def find_target_file(dir, file_name):
    all_files_recur = glob.glob(f'{dir}/**{file_name}*', recursive=True)
    path_target = None
    if len(all_files_recur) == 1:
        path_target = all_files_recur[0]

    assert not len(all_files_recur) > 1
    return path_target

def copy_files_in_dir(dir_src, dir_target, ext_file, rename_mode = 'stem'):
    '''Copy files in dir and rename it if needed
    '''
    ensure_dir_existence(dir_target)
    vec_path_files = sorted(glob.glob(f'{dir_src}/*{ext_file}'))
    for i in range(len(vec_path_files)):
        path_src = vec_path_files[i]

        if rename_mode == 'stem':
            pp, stem, _ = get_path_components(path_src)
            path_target = f'{dir_target}/{stem}{ext_file}'
        elif rename_mode == 'order':
            path_target = f'{dir_target}/{i}{ext_file}'
        elif rename_mode == 'order_04d':
            path_target = f'{dir_target}/{i:04d}{ext_file}'
        else:
            NotImplementedError

        copy_file(path_src, path_target)
    return len(vec_path_files)

# time-related funcs
def get_consumed_time(t_start):
    '''
    Return:
        time: seconds
    '''
    t_end = datetime.now()
    return (t_end-t_start).total_seconds()

def get_time():
    '''
    Return:
        time: seconds
    '''
    return datetime.now()


def get_time_str(fmt='HMSM'):
    if fmt == 'YMD-HMS':
        str_time = datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
    elif fmt == 'Y_M_D-H_M_S':
        str_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    elif fmt == 'Y_M_D':
        str_time = datetime.now().strftime("%Y_%m_%d")
    elif fmt == 'MDY-HMS':
        str_time = datetime.now().strftime("%m/%d/%Y-%H:%M:%S")
    elif fmt == 'HMS':
        str_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    elif fmt == 'HMSM':
        str_time = datetime.now().strftime("%H_%M_%S_%f")
    return str_time

def write_dict_to_h5(path, dict_data, should_ensure_pdir_existence=False):
    '''Write data to h5 file
    '''
    if should_ensure_pdir_existence:
        ensure_pdir_existence(path)
    f1 = h5py.File(path, 'w')
    for key in dict_data:
        if dict_data[key] is not None:
            f1.create_dataset(key, data=dict_data[key], compression='gzip', compression_opts=4)
    f1.close()

def read_h5_to_dict(path, use_tensor=False, should_expand_tensor=False, keys_to_skip=[], keys_skip2tensor=[]):
    '''Write data to h5 file
    '''
    f1 = h5py.File(path, 'r')
    dict_data = {}
    for key in f1:
        if key in keys_to_skip:
            continue
        # print(key)
        dict_data[key] = f1[key][:]
        if use_tensor and (key not in keys_skip2tensor):
            dict_data[key] = torch.from_numpy(dict_data[key])
            if should_expand_tensor:
                dict_data[key] = dict_data[key][None, ...]

        if should_expand_tensor:
            if (key in keys_skip2tensor):
                # for list of string
                dict_data[key] = [dict_data[key]]
            elif not use_tensor:
                dict_data[key] = dict_data[key][None, ...]

    f1.close()
    return dict_data

def write_dict_to_h5_recursively_inside(h5_group, data, check_existence_item=False):
    for key, value in data.items():
        if isinstance(value, dict):
            if key not in h5_group:
                subgroup = h5_group.create_group(key)
            else:
                subgroup = h5_group[key]
            write_dict_to_h5_recursively_inside(subgroup, value)
        else:
            if key in h5_group:
                if check_existence_item:
                    continue
                else:
                    del h5_group[key]
            if isinstance(value, float) or isinstance(value, int) or isinstance(value, str):
                h5_group.create_dataset(key, data=value)
            else:
                h5_group.create_dataset(key, data=value, compression='gzip', compression_opts=4)

def write_dict_to_h5_recursively(path, dict_data, should_ensure_pdir_existence=True, mode_write='w'):
    '''Write data to h5 file recursively'''
    if should_ensure_pdir_existence:
        ensure_pdir_existence(path)

    with h5py.File(path, mode_write) as h5_group:
        write_dict_to_h5_recursively_inside(h5_group, dict_data)

def read_h5_to_dict_recursively_inside(h5_group, lis_key_to_load=None):
    data = {}
    if lis_key_to_load is None:
        lis_key_to_load = h5_group.keys()

    for key in lis_key_to_load:
        item = h5_group[key]
        if isinstance(item, h5py.Group):
            data[key] = read_h5_to_dict_recursively_inside(item)
        else:
            data[key] = item[()] #item[()]
    return data

def read_h5_to_dict_recursively(path, lis_key_to_load=None):
    # Load the data from the HDF5 file
    # with h5py.File(path, "r") as h5file:
    #     loaded_data = read_h5_to_dict_recursively_inside(h5file)

    h5file = h5py.File(path, "r")
    loaded_data = read_h5_to_dict_recursively_inside(h5file, lis_key_to_load=lis_key_to_load)
    h5file.close()

    return loaded_data

def read_h5_tsdf_to_dict(path_h5):
    dict_data = read_h5_to_dict(path_h5)
    if 'mask_valid' in dict_data:
        thres_trunc = dict_data['thres_trunc'][0]
        mask_sdf = dict_data['mask_valid']
        sdf_np = np.ones_like(mask_sdf).astype(np.float32) * thres_trunc
        sdf_np[mask_sdf] = dict_data['tsdf_valid']
        sdf = sdf_np
        dict_data['tsdf'] = sdf
        n_elem = sdf.shape[0]

    if n_elem==(128**3):
        tsdf_reso = 128
    if n_elem==(256**3):
        tsdf_reso = 256
    else:
        assert False, 'Not implemented'
    dict_data['tsdf'] = dict_data['tsdf'].reshape(tsdf_reso,tsdf_reso, tsdf_reso)
    return dict_data, tsdf_reso

def write_list_to_txt(path_list, data_list, mode_write='w'):
    ensure_pdir_existence(path_list)
    num_lines = len(data_list)
    with open(path_list, mode_write) as flis:
        for i in range(len(data_list)):
            flis.write(f'{data_list[i]}\n')

def write_dict_to_json(path, dict, mode_write='w', check_existence=True):
    if check_existence:
        ensure_pdir_existence(path)
    # Serializing json
    json_object = json.dumps(dict, indent=4)

    # Writing to sample.json
    with open(path, mode_write) as outfile:
        outfile.write(json_object)

def read_json(path_json):
    with open(path_json) as json_file:
        data = json.load(json_file)
    return data

def read_json_to_dict(path_json):
    with open(path_json) as json_file:
        data = json.load(json_file)

    data = OrderedDict(sorted(data.items()))
    return data

def get_gpu_usage():
    # https://blog.csdn.net/weixin_49731222/article/details/123051521
    gpu = GPUtil.getGPUs()[0]
    # print(gpu.memoryUsed, gpu.memoryUtil * 100, gpu.load * 100)
    return gpu

def get_img_name(vec_path):
    '''merge img name of partnet name
    '''
    vec_comp_img_name = []
    count_name = 0
    for j in range(len(vec_path)):
        # if j == 0:
        #     print(data_dict["path"][j])
        comp_currr = vec_path[j].split('/')
        if comp_currr[-3] not in vec_comp_img_name:
            vec_comp_img_name.append(comp_currr[-3])

        if comp_currr[-2] not in vec_comp_img_name:
            vec_comp_img_name.append(comp_currr[-2])

        vec_comp_img_name.append(comp_currr[-1][:2])

        count_name += 1
        if count_name > 7:
            break

    # concat comps
    img_name = vec_comp_img_name[0] + '/'
    for idx_comp in range(1, len(vec_comp_img_name)):
        img_name += vec_comp_img_name[idx_comp] + '_'
    img_name = img_name[:-1] + '.png'
    return img_name

####
def convert_arr_str_bin2ascii(arr_bin):
    arr_str = []
    for item in arr_bin:
        arr_str.append(item.decode('ascii'))
    return arr_str

def convert_lis2dict(lis_data):
    dict_data = {}
    for i in range(len(lis_data)):
        dict_data[str(i)] = lis_data[i]
    return dict_data

def convert_lis_binary2ascii(lis_binary):
    lis_ascii = []
    for i in range(len(lis_binary)):
        lis_ascii.append(lis_binary[i].decode('ascii'))
    return lis_ascii


def get_tgt_files(dir_mesh, ext_file):
    lis_files = []
    for root, dirs, files in os.walk(dir_mesh):
        # for dir in dirs:
        #     if 'example' not in dir:
        #         continue

        for file in files:
            if file.endswith(ext_file):
            #     if 'level_' in file and 'level_0' not in file:
            #         continue
            # # if file.endswith('.ply') and 'mesh_gt' in file:
                lis_files.append(os.path.join(root, file))
    return lis_files

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=sys.stdout,
        filename='example.log'
    )
