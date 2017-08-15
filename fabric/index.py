#!/usr/bin/env python
# -*- coding: utf-8 -*-

import web
import json
import caffe_model_PROJECT as caffe_model
import pickle
from resize_img import resize
import os

rootUrl = '/validfileupload'
urls = (rootUrl+'/PROJECT', 'recognizer')

def recognize(model):
    response = {'errno':0, 'cate':'', 'result':''}
    try:
        response['result'] = model.predict(web.webapi.rawinput())
    except Exception,ex:
        response['errno'] = str(ex)

    return json.dumps(response,ensure_ascii=False)


class recognizer:
    def POST(self):
        return recognize(caffe_model)
    def GET(self):
        return 'Hello, world!'

app = web.application(urls, globals())


if __name__ == "__main__":
    web.wsgi.runwsgi = lambda func, addr=None: web.wsgi.runfcgi(func, addr)
    app.run()
