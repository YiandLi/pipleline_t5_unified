# -*- coding: utf-8 -*-
import logging

from petrel_client.client import Client

LOG = logging.getLogger('petrel_client.test')
file_url = 's3://wiki/data/wiki_test/'
file_name = '00054edbef3bfdaa524db1d7ea6de454.txt'

conf_path = '~/petreloss.conf'
client = Client(conf_path)

# 读取
txt_bytes = client.get(file_url + file_name)
assert (txt_bytes is not None)
txt_bytes = memoryview(txt_bytes)

print(txt_bytes)
