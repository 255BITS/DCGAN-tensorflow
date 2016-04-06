import numpy as np
import convert_raw_data
import math
import tensorflow_wav
def leaf(t, data):
    jump = 2 # todo, different on different db modes 
    levels = len(data)
    last_level = data[levels-1][t*jump:(t+1)*jump]
    path = []
    current_level = levels-2
    index = (t*jump)
    while(current_level >= 0):
        index //=2
        cdata = data[current_level]
        path.append(cdata[index])
        current_level-=1

    path.reverse()
    
    for elem in last_level:
        path.append(elem)

    return np.reshape(np.array(path), -1)


def tree_from(leaf):
    tree = []
    for elem in leaf[:-2]:
        tree.append([elem])
    tree.append([leaf[-2], leaf[-1]])
    return tree



def tree_merge(tree, tree2):
    if(tree == []):
        return tree2
    last_level = 1
    # todo combine leaf with tree


def tree_from_dense(dense):
    layers = []
    for n,layer in enumerate(dense):
        max = 2**(n-1)
        #print('max is', max)
        length = len(layer)
        layer = np.array(layer)
        print("length is ", length, max, n)
        layers.append(layer[0::math.ceil(length/max)].tolist())
    return layers
def reconstruct_tree(leaves, tree=[]):
    tree_total = []
    layers = [[] for _ in range(len(tree_from(leaves[0])))]
    for leaf in leaves:
        tree = tree_from(leaf)
        for layer_i in range(len(tree)):
            layers[layer_i].append(tree[layer_i])
            #tree_total[layer_i]=tree_total[layer_i]+tree[layer_i]
            #print("Layer", layer_i, tree_total)

    stacked = [ np.hstack(layer) for layer in layers ]

    return tree_from_dense(stacked)


    

def leaves_from(audio):
    leaves = []
    n=0
    sum=0
    total = len(audio[-1])
    for i in range(0, len(audio)):
        sum+=len(audio[i])
        print(i, len(audio[i]), sum)

    while(n<total//2):
        newleaf = leaf(n, audio)
        leaves.append(newleaf)
        #print(n,total//2)
        n+=1
    return leaves
        
if __name__ == '__main__':
    test_tree = [[0], [0],[0,1],[1,2,3,4],[3,4,5,6,7,8,9,10]]
    print("Input", test_tree)
    leaves = []
    leaves.append(leaf(0,test_tree))
    leaves.append(leaf(1,test_tree))
    leaves.append(leaf(2,test_tree))
    leaves.append(leaf(3,test_tree))

    print(leaf(0,test_tree))
    print(tree_from(leaves[0]))
    print(leaf(1,test_tree))
    print(tree_from(leaves[1]))
    print(leaf(2,test_tree))
    print(tree_from(leaves[2]))
    print(leaf(3,test_tree))
    print(tree_from(leaves[3]))

    print(" 0 leaf ",leaves[0])

    print(reconstruct_tree(leaves))


    input_wav = 'input.wav'
    processed = convert_raw_data.preprocess(input_wav)
    mlaudio_orig = tensorflow_wav.get_pre(input_wav+".mlaudio")

    mlaudio= mlaudio_orig["wavdec"]
    leaves = leaves_from(mlaudio[0])
    leaves_right = leaves_from(mlaudio[1])
    audio = reconstruct_tree(leaves)
    audio_right = reconstruct_tree(leaves_right)
    mlaudio = [audio, audio_right]
    print('mlaudio is', np.shape(mlaudio))
    mlaudio_orig['wavedec'] = mlaudio
    out = tensorflow_wav.convert_mlaudio_to_wav(mlaudio_orig)
    outfile = input_wav+".sanity.wav"
    tensorflow_wav.save_wav(out, outfile)


