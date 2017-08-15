import json
import time
import os
from fabric.api import *
import fabric.contrib.files as fabfiles
__author__ = 'xdyang'

captcha_servers = ['123.57.45.7']#, 'server2']
env.user = 'work'
env.password = 'yjyqhyxxcyshys'
env.roledefs = {
    'captcha_servers' : captcha_servers
}

from fab_config import config
@roles('captcha_servers')
def deploy():

    for key in config['local']:
       remote_path = config['remote'][key]
       remote_dir = os.path.dirname(remote_path)
       run('mkdir -p {0}'.format(remote_dir))
       put(config['local'][key], config['remote'][key])
       if key is 'index_py':
           run('chmod a+x {0}'.format(config['remote'][key]))
