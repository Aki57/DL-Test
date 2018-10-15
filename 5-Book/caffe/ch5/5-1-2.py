import sys
import caffe
from caffe.proto import caffe_pb2
import leveldb
import numpy as np
from skimage import io

def leveldb_process(path):
	db = leveldb.LevelDB(path)
	datum = caffe_pb2.Datum()

	item_id = 0
	for key,value in db.RangeIter():
    	datum.ParseFromString(value)
    	label = datum.label
    	data = caffe.io.datum_to_array(datum)
    	# do something here
    	item_id += 1
	print item_id
    
if __name__ == '__main__':
	path = sys.argv[1]
    leveldb_process(path)