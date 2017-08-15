

mkdir -p fabric
cp /data1/kevin.H/base_model/fabric/* fabric
project=$1
sed -i s/PROJECT/${project}/g fabric/fab_config.py
sed -i s/PROJECT/${project}/g fabric/index.py
sed -i s/PROJECT/${project}/g fabric/caffe_model.py

width=`cat config.py |grep 'width' | awk -F ":" '{print $2}'|sed s/,//g |sed s/\ //g`
height=`cat config.py |grep 'height' | awk -F ":" '{print $2}'|sed s/,//g |sed s/\ //g`
num_chars=`cat config.py |grep 'number_of_characters' | awk -F ":" '{print $2}'|sed s/,//g |sed s/\ //g`

echo ${width} ${height} ${num_chars}
sed -i s/WIDTH/${width}/g deploy.prototxt
sed -i s/HEIGHT/${height}/g deploy.prototxt
sed -i s/WIDTH/${width}/g fabric/fab_config.py
sed -i s/HEIGHT/${height}/g fabric/fab_config.py
sed -i s/NUM_CHARS/${num_chars}/g fabric/fab_config.py

echo spawn-fcgi -d ./  -f ./index_${project}.py -a 127.0.0.1 -p 10048
echo http://123.57.45.7:8080/validfileupload/${project}
