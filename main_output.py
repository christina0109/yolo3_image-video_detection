##images_test
# -*- coding: utf-8 -*- 
import json
import os
import yolo3_images
import yolo3_video
from flask import Flask, request, redirect, url_for,jsonify
from werkzeug.utils import secure_filename
from flask_restful import reqparse, abort, Api, Resource
from flask import Response, json,Request
import time 
import uuid
import shutil
import os
from PIL import Image
from ffmpy3 import FFmpeg
import subprocess
import ffmpeg
import pytesseract
from PIL import Image
import urllib 

UPLOAD_FOLDER = '/data/yolo/'
# ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'dcm', 'gif'])


app = Flask(__name__)
app.config['UPLOAD_FORDER'] = UPLOAD_FOLDER

#@app.route("/", methods=['GET'])
#def test():
#    return "test"


@app.route("/yolo3_images", methods=['GET', 'POST'])
def upload_fileimage():
    print(os.getcwd())
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        f = request.files['file']

        ext = f.filename.rsplit('.',1)[1]
        # print(ext)
        uuid_str = uuid.uuid4().hex
        new_name = uuid_str + '.' +ext

        savefile='test_images'
        path=UPLOAD_FOLDER+savefile

        os.chdir(path)
        # dir_list = os.listdir(file_dir)
        # print(os.getcwd())
        mon=time.strftime("%Y%m")

        if not os.path.exists(mon):
            os.makedirs(mon)

        path_fina = path + '/' +mon   
        # print(path_fina)   

            
        from unicodedata import normalize
        

        f.save(path_fina + '/' + secure_filename(
            normalize('NFKD', new_name).encode('utf-8', 'strict').decode('utf-8')))
        os.chdir(UPLOAD_FOLDER)
        print(os.getcwd())
        outcome=yolo3_images.draw_save(path_fina + '/' +new_name)
        
        shutil.move(outcome[0],path_fina)
        picture=outcome[0].split('/')[-1]
        print(picture)

        dict_path={}
        dict_path['original']='/host'+path_fina + '/' +new_name
        dict_path['analyse'] = '/host'+path_fina +'/'+ picture

        return jsonify(dict_path)

@app.route("/yolo3_video", methods=['GET', 'POST'])
def upload_filevideo():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        f = request.files['file']

        ext = f.filename.rsplit('.',1)[1]
        # print(ext)
        uuid_str = uuid.uuid4().hex
        new_name = uuid_str + '.' +ext

        savefile='test_video'
        path=UPLOAD_FOLDER+savefile

        os.chdir(path)

        mon=time.strftime("%Y%m")

        if not os.path.exists(mon):
            os.makedirs(mon)

        path_fina = path + '/' +mon      

        print(path_fina)
        from unicodedata import normalize
        

        f.save(path_fina + '/' + secure_filename(
            normalize('NFKD', new_name).encode('utf-8', 'strict').decode('utf-8')))
        # print(f.filename)
        os.chdir(UPLOAD_FOLDER)
        outcome1=yolo3_video.draw_save(path_fina + '/' +new_name)


        shutil.move(outcome1,path_fina)
        video=outcome1.split('/')[-1]
        print(video)


        path_old=savefile+'/'+mon +'/'+ video
        print(path_old)
        path_new=savefile+'/'+mon +'/'+uuid.uuid4().hex+'.mp4'
        print(path_new)
        getmp3='ffmpeg' +' -i '+path_old+' -c h264 '+path_new

        returnget= subprocess.call(getmp3,shell=True)

        os.remove(path_fina+'/'+video)

        dict_path={}

        dict_path['original']='/host'+path_fina + '/' +new_name
        dict_path['analyse'] = '/host'+UPLOAD_FOLDER+path_new


        return jsonify(dict_path)
    return jsonify("video")


@app.route("/word_analysis", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        f = request.files['file']
        ext = f.filename.rsplit('.',1)[1]
        # print(ext)
        uuid_str = uuid.uuid4().hex
        new_name = uuid_str + '.' +ext

        savefile='word_analysis/'

        path_fina=UPLOAD_FOLDER+savefile

        from unicodedata import normalize


        f.save(path_fina  + secure_filename(
            normalize('NFKD', new_name).encode('utf-8', 'strict').decode('utf-8')))
        print(path_fina + new_name)

        image = Image.open(path_fina + new_name)

        code = pytesseract.image_to_string(image, lang='chi_sim')
        # outcome = bleeding.draw_image(UPLOAD_FOLDER + '/' + new_name)
        os.remove(path_fina  + new_name)

        dict_result={}
        dict_result['result']=code

        return jsonify(dict_result)

    return "word_analysis\n".title()


if __name__=="__main__":
    app.run(host='0.0.0.0',port='8888',debug=True)







    
