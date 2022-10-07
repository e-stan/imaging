import SIMSIToolBox
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pickle as pkl
import molmass
import SIMSIToolBox.recalibration


datadir = "X:/Kevin/Bruker/MALDI/imzML/"
subdirs = {"50um":"092622_Brain_C12_50um/","100um":"092622_Brain_C12_100um/"}

ppmThresh = 5
num_cores = 20
intensityCutoff = 100
convSquare = 3 #size of filter (1=1x1,3=3x3,5=5x5)
colormap = LinearSegmentedColormap.from_list("test",colors=["black", "navy","blue","cyan","yellow","orange","orangered","red","mistyrose","white"],N=256)
dm_method = "PCA" #method for dimensionality reduction ("PCA" or "TSNE") PCA has worked better for me
seg_method = "K_means" #thresholding method ("TIC_auto", "K_means", "TIC_manual")
num_components = 2 #number of compoents to use with PCA or TSNE
filt = "GB" #filtering method (GB = gaussian blur, MA = moving average)

peaklist = pd.read_csv(datadir + "metabolites_with_signal_to_extract.csv")

if __name__ == "__main__":
    for res in subdirs:
        files = [x for x in os.listdir(datadir + subdirs[res]) if "_recal.imzML" in x]
        for fn in files:
            #read in peak list
            #peaklist = pd.read_csv(datadir + subdirs[res] + "metaspace_annotations_" + fn.replace(".imzML", ".csv"), header=2)


            #gather mzs of interest for all metaspace metabolites and isotopes
            mzs = []
            keys = []
            for index,row in peaklist.iterrows():
                _,_,nC = SIMSIToolBox.getMzsOfIsotopologues(row["formula"],"C")
                for x in range(nC+1):
                    mzs.append(row["mz"] + 1.00336 * x)
                    keys.append((index,x))

            #read in data, segment, and smooth
            msi = SIMSIToolBox.MSIData(mzs, ppm=ppmThresh, numCores=num_cores, intensityCutoff=intensityCutoff)
            msi.readimzML(datadir + subdirs[res] + fn)
            msi.segmentImage(method=seg_method, num_latent=num_components, dm_method=dm_method, fill_holes=True)
            msi.smoothData(filt,convSquare)

            #run natural abundance correction filtering out incompatible metabolites
            inds = []
            toDrop = []
            for index, row in peaklist.iterrows():
                tmp = [x for x in range(len(keys)) if keys[x][0] == index]
                tmp.sort(key=lambda x: keys[x][1])
                inds.append(tmp)
                f = molmass.Formula(row["formula"])  # create formula object
                comp = f.composition()
                bad = False
                for row in comp:
                    if row[0] not in ["H", "C", "N", "O", "S", "P", "Si"]:
                        bad = True
                        break
                if bad:
                    toDrop.append(index)
            peaklist["inds"] = inds
            peaklist = peaklist.drop(toDrop, axis=0)
            msi.correctNaturalAbundance(peaklist["formula"].values, peaklist["inds"].values)

            #score the M0 image and save result
            scores = []
            for index, row in peaklist.iterrows():
                tmp = msi.data_tensor[row["inds"]]
                isoTensor = SIMSIToolBox.normalizeTensor(tmp)
                scores.append(np.mean(isoTensor[0][msi.imageBoundary > 0.5]))
            peaklist["meanM0"] = scores
            peaklist = peaklist.sort_values(by="meanM0", ascending=False)

            #output processed data
            #df = msi.to_pandas()
            pkl.dump([peaklist, msi], open(datadir + subdirs[res] + fn.replace(".imzML", "_corrected_goodMets.pkl"), "wb"))

            plt.figure()
            plt.imshow(np.mean(msi.mass_errors, axis=0))
            plt.colorbar()
            plt.title(fn)

    plt.show()