from SIMSIToolBox import *
import os
import h5py
from pyimzml.ImzMLWriter import  ImzMLWriter

datadir = "../data/Davidson_data/"

for file in os.listdir(datadir):

    if ".mat" in file:
        print(file)
        f = h5py.File(datadir + file,"r")
        dat = f["msi"]["data"]
        ind = 0
        output = ImzMLWriter(datadir + file.replace(".mat",".imzML"), polarity='positive',mode="processed")

        ids = []
        tmp = dat["id"]
        for id in tmp:
            ids.append(np.array(f[id[0]])[0][0])

        xs = []
        tmp = dat["x"]
        for id in tmp:
            xs.append(np.array(f[id[0]])[0][0])

        ys = []
        tmp = dat["y"]
        for id in tmp:
            ys.append(np.array(f[id[0]])[0][0])

        for id,x,y,i in zip(ids,xs,ys,range(len(xs))):
            mzs = np.array(f[dat["peak_mz"][i][0]])[:,0]
            sigs = np.array(f[dat["peak_sig"][i][0]])[:,0]
            output.addSpectrum(mzs,sigs,(x,y))

        output.close()
        f.close()



