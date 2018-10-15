import numpy as np
import lmdb
import sys
import caffe
from caffe.proto import caffe_pb2

def lmdb_process(db_path):
    env = lmdb.open(db_path)
    datum = caffe_pb2.Datum()
    item_id = 0
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            datum.ParseFromString(value)
            label = datum.label
            img = caffe.io.datum_to_array(datum)
            # do something here
            item_id += 1
    print item_id

if __name__ == '__main__':
    db_path = sys.argv[1]
    lmdb_process(db_path)