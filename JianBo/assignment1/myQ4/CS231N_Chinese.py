#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by wjbKimberly on 17-12-12

import re
import urllib.request
import os


doc_url = "http://vedio.leiphone.com/5a28f6d9006e2"
file_name=15



ts_loc="/home/wjb/cs231n/CS231N_2017_Chinese/%s/"%file_name
url_head="http://vedio.leiphone.com"
mp4_loc = "/home/wjb/cs231n/CS231N_2017_Chinese/"


def get_ts_series():
    html_sample = str(urllib.request.urlopen(doc_url).readlines())
    patt=r"b'(/.+?\.ts)\\n"
    url_list=re.findall(patt,html_sample)
    print(url_list)

    for ui in url_list:
        ui=url_head+ui
        patt_name=r"\d{6}.ts"
        namei=re.findall(patt_name,ui)[0]
        urllib.request.urlretrieve(ui, ts_loc+'%s' % namei)

def convert_ts_2_mp4():
    content = ""
    lists = os.listdir(ts_loc)
    lis=sorted(lists)

    cmd = "cd %s && ffmpeg -i \"concat:"%mp4_loc
    for file in lis:
        if file != '.DS_Store':
            file_path = os.path.join(ts_loc, file)
        cmd += file_path + '|'
        # print("文件：%s"%file_path)
    cmd = cmd[:-1]
    cmd += '" -bsf:a aac_adtstoasc -c copy -vcodec copy %s.mp4'%file_name
    try:
        os.system(cmd)
        content += "file '%s.mp4'\n"%file_name
    except:
        print("Unexpected error")

    fp = open("%smp4list.txt"%mp4_loc,'a+')
    fp.write(content)
    fp.close()
    mp4cmd = "cd %s && ffmpeg -y -f concat -i mp4list.txt"%(mp4_loc)
    os.system(mp4cmd)


if __name__ == '__main__':
    get_ts_series()
    convert_ts_2_mp4()