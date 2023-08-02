import re
import os

# Get the absolute path of the current project
main_dir = os.path.dirname(os.path.abspath(__file__))

def read_blosum_aa(path):
    '''
    Read the blosum matrix from the file blosum50.txt
    Args:
        1. path: path to the file blosum50.txt
    Return values:
        1. The blosum50 matrix
    '''
    with open(path,"r") as f: 
        blosums = []
        aa={}
        index = 0
        for line in f:
            blosum_list = []
            line = re.sub("\n","",line)
            for info in re.split("\s+",line):
                try:
                    blosum_list.append(float(info))
                except:
                    if info not in aa and info.isalpha():
                        aa[info] = index
                        index += 1
            if len(blosum_list) > 0:
                blosums.append(blosum_list)
    assert (len(blosums[0]) == len(i) for i in blosums)   
    return blosums,aa


# Obtain BLOSUMS62 matrix (for protein mapping to numerical matrix) and AA (amino acids with their labels)
BLOSUMS,AA = read_blosum_aa(main_dir + r"/data/blosum.txt")
