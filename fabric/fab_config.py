



config = {
    'width':WIDTH,
    'height':HEIGHT,
    'num_chars':NUM_CHARS,
    'local':{
        'proto':'../deploy.prototxt',
        'model':'../model/PROJECT_iter_30000.caffemodel',
        'rdict':'/data1/kevin.H/base_model/dict/revert_alphanum_dict.pkl',
        'index_py':'index.py',
        'caffe_model_py':'caffe_model.py',
        'config': 'fab_config.py'
    },
    'remote':{
        'proto': '/home/work/xdyang/validcode_server_150727/res/proto/PROJECT/deploy.prototxt',
        'model': '/home/work/xdyang/validcode_server_150727/res/model/PROJECT/deploy.model',
        'rdict': '/home/work/xdyang/validcode_server_150727/res/dict/PROJECT/revert_alphanum_dict.pkl',
        'index_py': '/home/work/xdyang/validcode_server_150727/index_PROJECT.py',
        'caffe_model_py':'/home/work/xdyang/validcode_server_150727/caffe_model_PROJECT.py',
        'config':'/home/work/xdyang/validcode_server_150727/config/PROJECT.py',
    }
}

