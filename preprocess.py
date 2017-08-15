import cv2
import os
import h5py
import time
import config
import random
import pickle
import re
import numpy as np
import commands

__author__ = 'xdyang'

# match -timestamp in file name
ptn = re.compile(r"_\d+\.")

file_md5 = {}

def uniq_md5(file):
    res = commands.getoutput('md5sum {}'.format(file))
    md5 = res.split(' ')[0]
    if file_md5.has_key(md5):
        return False
    else:
        file_md5[md5] = 0
        return True
# we assume the file name's format is label-timestamp.extesion. Example: 0-1437718807817.jpg
def get_label(file_name):
    match = ptn.search(file_name)
    if not match:
        return None
    start, end = match.span()
    label = file_name[:start]
    return label


def is_valid_file(file_name):
    label = get_label(file_name)
    if label is None:
        return False
    if len(label) != global_validCode_alphanumber_length:
        return False
    return True


def is_valid_size(file_name):
    try:
        img = cv2.imread(file_name)

        width = img.shape[1]
        height = img.shape[0]
        if width != global_width or height != global_height:
            #print width,global_width,height,global_height
            return False
        else:
            return True
    except:
        return False

def is_valid_arithmetic_operation_label(file_name):
    label = get_label(file_name)
    if label is None:
        return False
    try:
        value = int(label)
        if -global_validCode_number_range < value < global_validCode_number_range:
            return True
        else:
            return False
    except:
        return False


def is_valid_alpha_number_string_label(file_name):
    label_len = global_validCode_alphanumber_length
    label = get_label(file_name)
    if label is None:
        return False
    if len(label) == label_len and label.isalnum():
        return True
    else:
        return False

def is_valid_chinese_phrase_label(file_name):
    label = get_label(file_name)
    if label is None:
        return False
    ustr = label.decode('utf-8')
    for uchar in ustr:
        if uchar < u'\u4e00' or uchar > u'\u9fa5':
            return False
    return True

def gen_img_list_file(file_list, label_dict, out_dir, mode="single-label", out_file_name="filelist"):
    with open(os.path.join(out_dir, out_file_name), "w") as fout:
        for file_name in file_list:
            raw_label = get_label(file_name)
            try:
                if mode == "single-label":
                    label = [str(label_dict[raw_label])]
                else:
                    label = [str(label_dict[x.lower()]) for x in raw_label]
            except:
                print "Warning: label not found in dictionary: " + raw_label

            fout.write(file_name+" " + " ".join(label) + "\n")


# generate data in HDF5 format
# we can consider an HDF5 file as a dictionary
# if we have N images of size [height, width, channel] and k labels [label1, label2, .. labelk],
# then this function generates a dictionary with fields:
# data: a matrix of size [N, channel, height, width]
# and k fields named as label0, label1, .., labelk-1 each of which is an array of size N
def save_hdf5_data(lines, img_dir_path, hdf5_path, img_flag):
    img_list = []
    labels_list = []
   
    for line in lines:
        tokens = line.strip().split(' ')
        img_path = tokens[0]
        if img_dir_path is not None:
            img_path = os.path.join(img_dir_path, img_path)
        try:
            # print img_path
            img = cv2.imread(img_path, img_flag)
            # transpose image matrix from [height, width, channel] to [channel, height, width]
            img = img.transpose([2, 0, 1])
            # convert uint8 to float32
            img = img.astype('float32')
            img /= 255.0
            # reshape image matrix to a 4D matrix [1, channel, height, width]
            img = img.reshape([1]+list(img.shape))

            img_list.append(img)

            labels = np.array([int(x) for x in tokens[1:]], dtype='float32')
            labels_list.append(labels)
        except:
            print 'Ignore broken image: ' + img_path
    img_mat = np.vstack(img_list)

    labels_mat = np.vstack(labels_list)
    labels_mat = labels_mat.transpose()
    print img_mat.shape
    print labels_mat.shape
    print hdf5_path
    with h5py.File(hdf5_path, 'w') as fout:
        fout['data'] = img_mat
        for i in range(labels_mat.shape[0]):
            fout['label'+str(i)] = labels_mat[i, :]


def gen_hdf5_data(img_list_path,
                  img_dir_path=None,
                  hdf5_dir='data_hdf5',
                  file_prefix='hdf5_',
                  img_flag=cv2.IMREAD_COLOR,
                  cnt_per_batch=10000):
    hdf5_list_file_path = os.path.join(hdf5_dir, file_prefix + 'list.txt')
    hdf5_list = []
    file_cnt = 0
    with open(img_list_path) as fp:
        lines = []
        for i, line in enumerate(fp):
            if i % cnt_per_batch == 0 and i != 0:
                out_path = os.path.join(hdf5_dir, "%s%03d" % (file_prefix, file_cnt))
                save_hdf5_data(lines, img_dir_path, out_path, img_flag)
                hdf5_list.append(out_path)
                file_cnt += 1
                lines = []
            lines.append(line)
        out_path = os.path.join(hdf5_dir, "%s%03d" % (file_prefix, file_cnt))
        save_hdf5_data(lines, img_dir_path, out_path, img_flag)
        hdf5_list.append(out_path)
    with open(hdf5_list_file_path, 'w') as fp:
        fp.write('\n'.join(hdf5_list) + '\n')

# preprocess steps:
# filter image files that have valid label
# split file set into to subset: one for training, the other test
# for each data set, first generate files containing the image file paths and their converted labels
# then generate hdf5 data files and a file containing the hdf5 file paths
def prepare_data_hdf5(img_dir, list_file_dir, hdf5_root_dir, label_dict, check_func, shuffle=True):
    file_list = os.listdir(img_dir)
    file_list = [file_name for file_name in file_list if check_func(file_name) and is_valid_size(os.path.join(img_dir, file_name)) \
                 and uniq_md5(os.path.join(img_dir, file_name))]
    print len(file_list)  
    if shuffle:
        random.shuffle(file_list)

    if not os.path.exists(list_file_dir):
        os.makedirs(list_file_dir)
    if not os.path.exists(hdf5_root_dir):
        os.makedirs(hdf5_root_dir)

    train_cnt = int(len(file_list) * 0.9)
    meta_info = [('train', file_list[:train_cnt]), ('test', file_list[train_cnt:])]
    for info in meta_info:
        img_list = info[1]
        file_list_name = info[0]+".txt"
        gen_img_list_file(img_list, label_dict, list_file_dir, mode="multi-label", out_file_name=file_list_name)
        prefix = info[0] + '_hdf5_'
        gen_hdf5_data(os.path.join(list_file_dir, file_list_name), img_dir_path=img_dir, hdf5_dir=hdf5_root_dir, file_prefix=prefix)


def prepare_data_lmdb(img_dir, list_file_dir, lmdb_root_dir, label_dict, check_func, shuffle=True):
    file_list = os.listdir(img_dir)
    file_list = [file_name for file_name in file_list if check_func(file_name) and is_valid_size(os.path.join(img_dir, file_name))]
    if shuffle:
        random.shuffle(file_list)

    if not os.path.exists(list_file_dir):
        os.makedirs(list_file_dir)
    if not os.path.exists(lmdb_root_dir):
        os.makedirs(lmdb_root_dir)

    train_cnt = int(len(file_list) * 0.9)
    meta_info = [('train', file_list[:train_cnt]), ('test', file_list[train_cnt:])]
    for info in meta_info:
        img_list = info[1]
        file_list_name = info[0]+".txt"
        lmdb_dir = os.path.join(lmdb_root_dir, info[0])
        gen_img_list_file(img_list, label_dict, list_file_dir, mode="single-label", out_file_name=file_list_name)
        os.system("sh lmdb.sh %s %s %s" % (img_dir, os.path.join(list_file_dir, file_list_name), lmdb_dir))


def prepare_multi_type_data(img_dir, list_file_dir, lmdb_root_dir, shuffle=True):
    file_list = os.listdir(img_dir)
    file_list_with_category = []
    for file_name in file_list:
        #if not is_valid_size(is_valid_size(os.path.join(img_dir, file_name))): continue

        # if is_valid_chinese_phrase_label(file_name):
        #     file_list_with_category.append([file_name, "0"])#3
        # el
        if is_valid_arithmetic_operation_label(file_name):
            file_list_with_category.append([file_name, "1"])
        elif is_valid_alpha_number_string_label(file_name):
            file_list_with_category.append([file_name, "0"])
    if shuffle:
        random.shuffle(file_list_with_category)

    if not os.path.exists(list_file_dir):
        os.makedirs(list_file_dir)
    if not os.path.exists(lmdb_root_dir):
        os.makedirs(lmdb_root_dir)

    train_cnt = int(len(file_list_with_category) * 0.9)
    meta_info = [('train', file_list_with_category[:train_cnt]), ('test', file_list_with_category[train_cnt:])]
    for info in meta_info:
        sub_list_with_category = info[1]
        file_list_name = info[0]+".txt"
        file_list_path = os.path.join(list_file_dir, file_list_name)
        with open(file_list_path, 'w') as fp:
            for x in sub_list_with_category:
                fp.write(" ".join(x) + '\n')

        lmdb_dir = os.path.join(lmdb_root_dir, info[0])
        os.system("./lmdb.sh %s %s %s %s %s" % (img_dir, file_list_path, lmdb_dir, global_height, global_width))
def update_deploy_proto():
    os.system('sed -i \'s/WIDTH/{0}/g\''.format(config.task['width']))
    os.system('sed -i \'s/HEIGHT/{0}/g\''.format(config.task['height']))

if __name__ == '__main__':
   
    print config.task
    img_dir = config.task['image_dir']
    global_width = config.task['width']
    global_height = config.task['height']
    label_dict = pickle.load(open(config.task['dict']))
    hdf5_dir = os.path.join(config.task['workspace'], 'hdf5')
    list_file_dir = config.task['workspace']
    global_validCode_number_range = config.task.get('number_range', 400)
    global_validCode_alphanumber_length = config.task.get('number_of_characters', 4)
    ptn = re.compile(r"{0}\d+\.".format(config.task['separator']))
    prepare_data_hdf5( config.task['image_dir'], list_file_dir, hdf5_dir, label_dict, is_valid_alpha_number_string_label)
    update_deploy_proto()

    """
    global_width = 132
    global_height = 44

    global_validCode_number_range = 400
    global_validCode_alphanumber_length = 4


    dict_path = config.alphanum_dict_path
    label_dict = pickle.load(open(dict_path))
    list_file_dir =  os.getcwd()
    hdf5_dir = os.path.join(os.getcwd(), 'hdf5')
    prepare_data_hdf5('/data1/kevin.H/copyright/1.raw/', list_file_dir, hdf5_dir, label_dict, is_valid_alpha_number_string_label)
"""

    # img_dir = "/search/xdyang/data/validCode_jx/fangliukai/"
    # list_file_dir = "/search/xdyang/data/validCode_jx/filelist/"
    # lmdb_dir = "/search/xdyang/data/validCode_jx/lmdbdir/"
    # mydict = pickle.load(open(config.num_dict_path))
    # prepare_data_lmdb(img_dir, list_file_dir, lmdb_dir, mydict, is_valid_arithmetic_operation_label)

    # img_dir = "/search/zeyang/weixin/1.raw/"
    # list_file_dir = "/search/zeyang/weixin/2.captcha_preprocess/weixin_cat/"
    # lmdb_dir = "/search/zeyang/weixin/2.captcha_preprocess/weixin_cat"
    # mydict = pickle.load(open(config.num_dict_path))
    # prepare_multi_type_data(img_dir, list_file_dir, lmdb_dir)

    #print is_valid_size("/search/xdyang/data/validCode_sc/raw/0-1437718807817.jpg")
