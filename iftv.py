import sys
import math  
import random
import datetime
import numpy as np
import pandas as pd
import warnings
import pysam
from numpy import *
from numba import njit
from scipy import special
from scipy.stats import norm
from scipy.stats import rv_continuous, gamma
from sklearn import preprocessing
from scipy import stats 

#function to read fasta file
def readFasta(filename):
    seq = ''
    fread = open(filename)
    #delete row one
    fread.readline()

    line = fread.readline().strip()
    while line:
        seq += line
        line = fread.readline().strip()
        
    return seq


#function to read readCount file , generate readCount array
def readRd(filename, seqlen):
    print(seqlen)
    readCount = np.full(seqlen, 0.0)
    samfile = pysam.AlignmentFile(filename, "rb")
    for line in samfile:
        if line.reference_name:
            chr = line.reference_name.strip('chr')
            if chr.isdigit():
                posList = line.positions
                readCount[posList] += 1
        
    return readCount


# dataSet is input data,t is number of trees,p is sub-sampling size  
def iForest(dataSet, t, p):  
    Forest = []  
    limit = np.ceil(np.log2(p))  
  
    for i in range(t):  
        dataSet_New = sample(dataSet, p)  

        attr_List = [0]  
        root = inNode()  
        dataSet_Newest = dataSet_New  
        iTree(root, dataSet_Newest, attr_List, 0, limit)  
        Forest.append(root)  
    return  Forest


# p is size of sample and p has a limit in dataSet  
# do with dataSet  
def sample(dataSet, p):  
    data = []  
    dataSet_len = len(dataSet)  
  
    j = []  
    for k in range(p):  
        j.append(random.randint(0, dataSet_len - 1))  
  
    for i in j:  
        data.append(dataSet[int(i)])
  
    return data  

  
class exNode(object):  
    def __init__(self, size=0):  
        self.size = size  
  
  
class inNode(object):  
    def __init__(self, splitAtt=-1, splitValue=-1, left=None, right=None, exNode=None):  
        self.splitAtt = splitAtt  
        self.splitValue = splitValue  
        self.left = left  
        self.right = right  
        self.exNode = exNode  
  
  
# establish a tree in limitation by l  
# dataSet is input data,e is current tree height,l is height limit  
def iTree_remove(dataSet, attr):  
    i = random.choice(attr)  
    maxV = np.max(dataSet)  
    minV = np.min(dataSet) 
    for v in dataSet:  
        if float(v[i]) > maxV:  
            maxV = float(v[i])  
        if float(v[i]) < minV:  
            minV = float(v[i])  
    r = int(random.uniform(minV, maxV))  
    dataSetB = filterData(dataSet, i, r, '>')  
    dataSetS = filterData(dataSet, i, r, '<')  
    return dataSetB, dataSetS, i, r  
  

def iTree(node, dataSet, attr, e, l):  
    if e >= l or len(dataSet) <= 1:  
        exNode_t = exNode()  
        exNode_t.size = len(dataSet)  
        node.exNode = exNode_t  
  
        return exNode_t  
    else:  
        dataSetB, dataSetS, i, r = iTree_remove(dataSet, attr)  
  
        if len(set([v[i] for v in dataSet])) == 1:  
            exNode_t = exNode()  
            exNode_t.size = len(dataSet)  
            node.exNode = exNode_t   
  
            return exNode_t  
  
        node.splitAtt = i  
        node.splitValue = r  
  
        node.right = inNode()  
        node.left = inNode()  
        iTree(node.left, dataSetS, attr, e + 1, l)  
        iTree(node.right, dataSetB, attr, e + 1, l)  
  
  
# r is random number which between minV and maxV , r can divide from dataSet  in attribute i  
def filterData(dataSet, i, r, k):  
    dataSetP = []  
    if k == '<':  
        for v in dataSet:  
            if float(v[i]) <= r:  
                dataSetP.append(v)  
    if k == '>':  
        for v in dataSet:  
            if float(v[i]) > r:  
                dataSetP.append(v)  
    return dataSetP  

 
# PathLength is according to data in x  
def PathLength(x, T, e):  
    if T.splitAtt == -1:  
        return e + c(T.exNode.size)  
    else:  
        i = T.splitAtt  
        if x <= T.splitValue:  
            return PathLength(x, T.left, e + 1)  
        else:  
            return PathLength(x, T.right, e + 1)  
  
  
def c(n):  
    if n > 2:  
        return 2 * (math.log(n - 1, math.e) + 0.5772156649) - (2 * n - 2) / float(n)
    elif n == 2:
        return 1
    
    return 0  
  

def myFunc(data):
    x = []
    global forestList
    for tree in forestList:
        x.append(PathLength(data, tree, 0))

    scores = np.power(2, -np.mean(x)/c(256))
    return scores


def prox_L1(step_size: float, x: np.ndarray) -> np.ndarray:
    """
    L1 proximal operator
    """
    return np.fmax(x - step_size, 0) - np.fmax(- x - step_size, 0)



def prox_tv1d(step_size: float, w: np.ndarray) -> np.ndarray:
    """
    Computes the proximal operator of the 1-dimensional total variation operator.

    This solves a problem of the form

         argmin_x TV(x) + (1/(2 stepsize)) ||x - w||^2

    where TV(x) is the one-dimensional total variation

    Parameters
    ----------
    w: array
        vector of coefficients
    step_size: float
        step size (sometimes denoted gamma) in proximal objective function

    References
    ----------
    Condat, Laurent. "A direct algorithm for 1D total variation denoising."
    IEEE Signal Processing Letters (2013)
    """

    if w.dtype not in (np.float32, np.float64):
        raise ValueError('argument w must be array of floats')
    w = w.copy()
    output = np.empty_like(w)
    _prox_tv1d(step_size, w, output)
    return output



@njit
def _prox_tv1d(step_size, input, output):
    """low level function call, no checks are performed"""
    width = input.size + 1
    index_low = np.zeros(width, dtype=np.int32)
    slope_low = np.zeros(width, dtype=input.dtype)
    index_up  = np.zeros(width, dtype=np.int32)
    slope_up  = np.zeros(width, dtype=input.dtype)
    index     = np.zeros(width, dtype=np.int32)
    z         = np.zeros(width, dtype=input.dtype)
    y_low     = np.empty(width, dtype=input.dtype)
    y_up      = np.empty(width, dtype=input.dtype)
    s_low, c_low, s_up, c_up, c = 0, 0, 0, 0, 0
    y_low[0] = y_up[0] = 0
    y_low[1] = input[0] - step_size
    y_up[1] = input[0] + step_size
    incr = 1

    for i in range(2, width):
        y_low[i] = y_low[i-1] + input[(i - 1) * incr]
        y_up[i] = y_up[i-1] + input[(i - 1) * incr]

    y_low[width-1] += step_size
    y_up[width-1] -= step_size
    slope_low[0] = np.inf
    slope_up[0] = -np.inf
    z[0] = y_low[0]

    for i in range(1, width):
        c_low += 1
        c_up += 1
        index_low[c_low] = index_up[c_up] = i
        slope_low[c_low] = y_low[i]-y_low[i-1]
        while (c_low > s_low+1) and (slope_low[max(s_low, c_low-1)] <= slope_low[c_low]):
            c_low -= 1
            index_low[c_low] = i
            if c_low > s_low+1:
                slope_low[c_low] = (y_low[i]-y_low[index_low[c_low-1]]) / (i-index_low[c_low-1])
            else:
                slope_low[c_low] = (y_low[i]-z[c]) / (i-index[c])

        slope_up[c_up] = y_up[i]-y_up[i-1]
        while (c_up > s_up+1) and (slope_up[max(c_up-1, s_up)] >= slope_up[c_up]):
            c_up -= 1
            index_up[c_up] = i
            if c_up > s_up + 1:
                slope_up[c_up] = (y_up[i]-y_up[index_up[c_up-1]]) / (i-index_up[c_up-1])
            else:
                slope_up[c_up] = (y_up[i]-z[c]) / (i-index[c])

        while (c_low == s_low+1) and (c_up > s_up+1) and (slope_low[c_low] >= slope_up[s_up+1]):
            c += 1
            s_up += 1
            index[c] = index_up[s_up]
            z[c] = y_up[index[c]]
            index_low[s_low] = index[c]
            slope_low[c_low] = (y_low[i]-z[c]) / (i-index[c])
        while (c_up == s_up+1) and (c_low>s_low+1) and (slope_up[c_up]<=slope_low[s_low+1]):
            c += 1
            s_low += 1
            index[c] = index_low[s_low]
            z[c] = y_low[index[c]]
            index_up[s_up] = index[c]
            slope_up[c_up] = (y_up[i]-z[c]) / (i-index[c])

    for i in range(1, c_low - s_low + 1):
        index[c+i] = index_low[s_low+i]
        z[c+i] = y_low[index[c+i]]
    c = c + c_low-s_low
    j, i = 0, 1
    while i <= c:
        a = (z[i]-z[i-1]) / (index[i]-index[i-1])
        while j < index[i]:
            output[j * incr] = a
            output[j * incr] = a
            j += 1
        i += 1
    return


@njit
def prox_tv1d_cols(stepsize, a, n_rows, n_cols):
    """apply prox_tv1d along columns of the matri a
    """
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_cols):
        _prox_tv1d(stepsize, A[:, i], out[:, i])
    return out.ravel()


@njit
def prox_tv1d_rows(stepsize, a, n_rows, n_cols):
    """apply prox_tv1d along rows of the matri a
    """
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_rows):
        _prox_tv1d(stepsize, A[i, :], out[i, :])
    return out.ravel()

'''def sigmoid(x):
    num = x.shape[0]
    y = np.full(num*1, 0.0) 
    for i in range(num):
        if (x[i] == 0):
            y[i] = 1/2
        elif (x[i] > 0):
            y[i] = (1/2) * (1+np.exp(-x[i]))
        y[i] = 1/(1+np.exp(-x[i]))
    return y'''

def loadData(x):
    dataMat = []
    num = len(x)
    print(num)
    for i in range(num):
        #print(x[i])
        #print(x[i][0])
        dataMat.append(x[i][0])
    return dataMat

def distEclud(x,y):
    return sqrt(power(y - x,2))

def randCent(dataMat,k):
    centroids = np.full(k*1, 0.0)
    minJ = min(dataMat)
    rangeJ = float(max(dataMat) - minJ)
    for i in range(k):
        centroids[i] = minJ + rangeJ * random.random()
    centroids = sorted(centroids)
    return centroids

def kMeans(dataMat,k,distMeas = distEclud,createCent = randCent):
    m = len(dataMat)
    clusterAssment = np.full(m*1, 0.0)
    centroids = createCent(dataMat,k)
    clusterChanged = True
    while clusterChanged == True:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j],dataMat[i])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i] != minIndex:
                clusterChanged = True
            clusterAssment[i] = minIndex
        #print(centroids)

        for cent in range(k):
            index = (clusterAssment == cent)
            #print(index)
            dataclust = []
            for i in range(m):
                if index[i]:
                    dataclust.append(dataMat[i])
                
            centroids[cent] = mean(dataclust)

    return clusterAssment


def calculate(x,bl):    
    
    y = x
    w0 = 0.95
    p = 2
    
    BL = np.full((x.shape),[bl])
    
    l2CN = np.full((x.shape),[0])
    loss = 100
    alldev = np.full(90*1, 0.0)
    n = 0
    
    for i in range(90):
        p = 2
        
        l0 = (x - (1 - w0) * BL) / w0
        l0 = loadData(x)
        l2 = kMeans(l0,p)
        l3 = []
        for i in range(len(l2)):
            l3.append([l2[i]])
       
        l1_error = y - (l3 * BL / p * w0 + BL * (1 - w0))
        dev = np.mean(np.abs(l1_error))
        alldev[n] = dev
        if (loss >= dev):
            loss = dev
            purity = w0
            ploidy = p
        #p -= 1
        w0 -= 0.01
        n += 1
        
    l2CN = (x - (1 - purity) * BL) / (purity * BL / p)
    l2CN = loadData(l2CN)
        
    return  l2CN

def calculate1(x,bl):
    
    y = x
    w0 = 0.95
    p = 2
    
    BL = np.full((x.shape),[bl])
    
    l2CN = np.full((x.shape),[0])
    loss = 100
    alldev = np.full(90*1, 0.0)
    n = 0
    
    for i in range(90):
        p = 2
        
        l0 = (x - (1 - w0) * BL) / w0
        l0 = loadData(x)
        l2 = kMeans(l0,p)
        l3 = []
        for i in range(len(l2)):
            l3.append([l2[i]])
       
        l1_error = y - (l3 * BL / p * w0 + BL * (1 - w0))
        dev = np.mean(np.abs(l1_error))
        alldev[n] = dev
        if (loss >= dev):
            loss = dev
            purity = w0
            ploidy = p
        #p -= 1
        w0 -= 0.01
        n += 1
    
        
    return  purity

def calculate2(x,bl):
    
    y = x
    w0 = 0.95
    p = 2
    
    BL = np.full((x.shape),[bl])
    
    l2CN = np.full((x.shape),[0])
    loss = 100
    alldev = np.full(90*1, 0.0)
    n = 0
    
    for i in range(90):
        p = 2
        
        l0 = (x - (1 - w0) * BL) / w0
        l0 = loadData(x)
        l2 = kMeans(l0,p)
        l3 = []
        for i in range(len(l2)):
            l3.append([l2[i]])
       
        l1_error = y - (l3 * BL / p * w0 + BL * (1 - w0))
        dev = np.mean(np.abs(l1_error))
        alldev[n] = dev
        if (loss >= dev):
            loss = dev
            purity = w0
            ploidy = p
        #p -= 1
        w0 -= 0.01
        n += 1
    
    return  ploidy

def main(params):
    # params list
    binLen = int(params[0])
    chrFile = params[1]
    chrName = chrFile.split('.')[0]
    rdFile = params[2]
    outputFile = params[3]
    statisticFile = params[4]
    treeNum = int(params[5])
    treeSampleNum = int(params[6])
    alpha = float(params[7])
    threshold = float(params[8])
    calculateFile = params[9]
    beforeFile = params[10]
    errorNum = 0.005
    seq = readFasta(chrFile)
    #The length of seq
    seqlen = len(seq)
    rd = readRd(rdFile, seqlen)
    #trick: make the length of seq can ben divide fully
    seq = seq[0: len(seq) // binLen * binLen]
    rd = rd[0: len(rd) // binLen * binLen]
    #The length of seq(after trick)
    seqlen = len(seq)

    #fillin n and N position
    for i in range(seqlen):
        if seq[i] not in ['a', 'A', 't', 'T', 'c', 'C', 'g', 'G']:
            rd[i] = 2500000
    rd = rd.astype(np.float64)

    #compress to 1/binLen 
    binValues = np.full(int(seqlen / binLen), 0.0)
    binValues = binValues.astype(np.float64)
    for i in range(int(seqlen / binLen)):
        binValues[i] = np.mean(rd[binLen * i : binLen * (i+1)])
        if np.isnan(binValues[i]):
           print(rd[binLen * i : binLen * (i+1)])

    index = (binValues < 2500)
    reverse_index = (binValues >= 2500)
    trains = binValues[index]

    #gcbias correct
    gcPercentage = np.full(int(seqlen / binLen), 0.0)
    gcPercentage = gcPercentage.astype(np.float)
    for i in range(int(seqlen / binLen)):
        partSeq = seq[binLen*i : binLen*(i+1)]
        gc = partSeq.count('C') + partSeq.count('c') + partSeq.count('g') + partSeq.count('G')
        at = partSeq.count('A') + partSeq.count('a') + partSeq.count('t') + partSeq.count('T')
        if((gc + at) != binLen):
            gcPercentage[i] = 0
        else:
            gcPercentage[i] = 1.0 * gc / (gc + at)

    gcPercentage = gcPercentage[index]
    gcPercentage = np.round(gcPercentage * 1000).astype(np.int64)

    bincount = np.bincount(gcPercentage)

    global_avg = np.mean(trains)

    for i in range(len(trains)):
        if bincount[gcPercentage[i]] < 2:
            continue
        mean = np.mean(trains[gcPercentage == gcPercentage[i]])
        #trains[i] = global_avg * trains[i] / mean
        if mean - errorNum < 0.01 and mean - errorNum > -0.01:
            continue
        trains[i] = (global_avg - errorNum) * trains[i] / (mean - errorNum)
    rv = norm(*norm.fit(trains))
    binMean = rv.stats()[0]
    print("binmean:",binMean)
    

    npMatrix = trains
    std_data = preprocessing.scale(npMatrix)
    data = pd.DataFrame(std_data)

    global forestList
    forestList = iForest(data.values, treeNum, treeSampleNum)
    scores = np.full(int(seqlen / binLen), 0.0)
    scores[index] = data.applymap(myFunc).values.squeeze()

    scores[reverse_index] = np.min(scores[index])

    #res = prox_tv1d(alpha, scores)
    #res_mean = np.mean(res)
    #res_var = np.std(res)
    #new_res = 1 - 0.5 * special.erfc(- (np.sqrt(0.5)) * (res - res_mean) / res_var)
    #results = new_res <= threshold

    res = prox_tv1d(alpha, scores)
    gamma_params = rv_continuous.fit(gamma,res,floc=0,scale = 1)
    print(gamma_params)
    results = (1 - gamma.cdf(res, gamma_params[0], loc = gamma_params[1], scale = gamma_params[2])) < threshold

    # Calculate mean RD across the genome by removing significant bins.
    normData = np.full(int(seqlen / binLen), 0.0)
    normData[index] = trains
    normData[reverse_index] = 0
    R_idx = (results == True)  
    R_data = normData[R_idx]
    R_RDmean = np.mean(R_data)
    print("length of R_data",len(R_data))
    print("R_RDmean:",R_RDmean)

    
    # leader 04/17
    #R_reverse_idx = (results == False)  
    #R_re_data = trains[R_reverse_idx]
    #R_False_data = normData[results == False]
    a = sum(trains)    #14027
    b = a - sum(R_data)   #176
    R_re_RDmean = b/(14027-176) 
    #print("length of R_data",len(R_re_data))
    print("R_re_RDmean:",R_re_RDmean)

    pre = False
    cur = False
    count = 0
    start = 0
    end = 0
    Seg_RD = 0
    output = open(outputFile, 'w')
    output.write("Chr\tStart_pos\tEnd_pos\tLength\tGain/Loss\tCopy_number\tseg_RD\n")

    maxRD_del = 0
    minRD_del = 100000
    for i in range(results.shape[0]):
        cur = results[i]
        if pre == False and cur == True:
            count += 1
            Seg_RD += normData[i]
            pre = cur
            start = i
        elif pre == True and cur == True:
            count += 1
            Seg_RD += normData[i]
        elif pre == True and cur == False:
            pre = cur
            end = i
            if count >= 1:
                Seg_RD = Seg_RD/count
                if  Seg_RD < R_RDmean:
                    if maxRD_del < Seg_RD:
                       maxRD_del = Seg_RD
                    if minRD_del > Seg_RD:
                       minRD_del = Seg_RD
            count = 0
            Seg_RD = 0
        else:
            continue

    pre = False
    cur = False
    count = 0
    start = 0
    end = 0
    Seg_RD = 0
    ## estimate tumor purity mu
    mu1 = 2*(R_RDmean - maxRD_del)/R_RDmean
    mu2 = 1 - minRD_del/R_RDmean
    mu = (mu1+mu2)/2
    print("mu:",mu)

    x_loss = []
    watchingdata = []
    
    quantity = 0
    
    
    for i in range(results.shape[0]):
        cur = results[i]
        if pre == False and cur == True:
            count += 1
            Seg_RD += normData[i]
            pre = cur
            start = i
        elif pre == True and cur == True:
            count += 1
            Seg_RD += normData[i]
        elif pre == True and cur == False:
            pre = cur
            end = i
            if count >= 1: 
                Seg_RD = Seg_RD/count
                watchingdata.append(Seg_RD)
                absoluteRD = (Seg_RD - R_RDmean*(1-mu))/mu
                CN = 2 * absoluteRD / R_RDmean
                quantity += 1
                if CN < 0:
                   CN = 0                
                if  Seg_RD < R_RDmean:
                    CN = round(CN)
                    if CN == 2:
                       CN = CN - 1
                    x_loss.append([Seg_RD])
                                         
                    output.write(chrName + "\t" + str(start * binLen + 1) + "\t" + str(end*binLen) + "\t" + str((end - start)*binLen) + '\tLoss'+ "\t" + str(CN)+ "\t" + str(Seg_RD)+ "\n")
                    print(chrName + "\t" + str(start * binLen + 1) + "\t" + str(end*binLen) + "\t" + str((end - start)*binLen) + '\tLoss'+ "\t" + str(CN)+ "\t" + str(Seg_RD)+"\t")
                else:
                    CN = round(CN)
                    if CN == 2:
                       CN = CN + 1
                     
                    output.write(chrName + "\t" + str(start * binLen + 1) + "\t" + str(end*binLen) + "\t" + str((end - start)*binLen) + '\tGain'+ "\t" + str(CN)+ "\t" + str(Seg_RD)+"\n")
                    print(chrName + "\t" + str(start * binLen + 1) + "\t" + str(end*binLen) + "\t" + str((end - start)*binLen) + '\tGain'+ "\t" + str(CN)+ "\t" + str(Seg_RD)+"\t")
            count = 0
            Seg_RD = 0
        else:
            continue
    
    output.close()

    x_loss = np.array(x_loss)
    watchingdata = np.array(watchingdata)

    print("quantity:",quantity)
    print(x_loss)
    ca = []
    ca = calculate(x_loss, R_RDmean)
    purity = calculate1(x_loss, R_RDmean)
    ploidy = calculate2(x_loss, R_RDmean)
    print(ca,len(ca))
    
    chr = []
    start = []
    end = []
    length = []
    stat = []
    cn = []
    absolutrRD = []
    abCN = []
    g = [8,5,4,1,1,0,0,0,0]
    ground = np.array(g)
    
    num = 0
    output = open(outputFile, 'r')
    next(output)
    
    for c in output.readlines():
        c_array = c.split("\t")
        chr.append(c_array[0])
        start.append(c_array[1])
        end.append(c_array[2])
        length.append(c_array[3])
        cn.append(c_array[5])
        
        '''if c_array[4] == "Loss":
            c_array[4] = ca[num]
            num+=1'''
        stat.append(c_array[4])
    #print(cn)
    stat_len = len(cn)  
    
    print("watchingdata:",watchingdata)
    for i in range(len(cn)):
        print("i:",i)
        print("absolute:",(watchingdata[i] - (1 - purity) * (R_re_RDmean)) / (purity))  
        absolutrRD.append((watchingdata[i] - (1 - purity) * (R_re_RDmean)) / (purity))
        print("abCN:",(absolutrRD[i] * ploidy / R_re_RDmean))
        abCN.append((absolutrRD[i] * ploidy / R_re_RDmean))
#     
    output = open(outputFile, 'w')
    output.write("Chr\tStart_pos\tEnd_pos\tLength\tGain/Loss\tseg_RD\tCopy_number\tabCN\n")
    for i in range(stat_len):
        output.write(chr[i] + "\t" + start[i] + "\t" + end[i] + "\t" + length[i] + "\t" + stat[i]+ "\t" +  str(watchingdata[i]) + "\t" + cn[i] +  "\t" +  str(abCN[i])  +"\n")
#         output.write('\n')
    output.close()    
    
    statisticFile = open(statisticFile, 'a')
    statisticFile.write(str(purity) + "\t" + str(ploidy)+ "\t" + str(mu) + "\n")
    statisticFile.close()

    calculateFile = open(calculateFile, 'a')
    for i in range(stat_len):
        calculateFile.write(str(abCN[i]) + "\t")
    calculateFile.write('\n')
    calculateFile.close() 

    beforeFile = open(beforeFile, 'a')
    for i in range(stat_len):
        beforeFile.write(cn[i] + "\t")
    beforeFile.write('\n')
    beforeFile.close()













