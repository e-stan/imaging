
import SIMSIToolBox
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from copy import deepcopy



datadir = "X:/MSI_Shared_Data/13CImagingManuscript/raw_data/imzmls/subset_data/"
datadir = "X:/Kevin/Bruker/MALDI/imzML/processed_data/"
fn = "20220816_01_80w69h_03C_mt_tumor_brain4-1_13C_20um Analyte 2_1.csv"
fn = "brain_c12_nedc_neg_40um.csv"

metabolites = ["C6H8NO5","C6H12O9P","C16H31O2","C20H31O2","C5H8NO4","C6H11O6"]
#metabolites = ["C6H12O9P"]

#metabolites = ["C5H8NO4"]

polarity="negative"
num_cores=20
ppmThresh = 50
colormap = LinearSegmentedColormap.from_list("test",colors=["black", "navy","blue","cyan","yellow","orange","orangered","red","mistyrose","white"],N=256)
filt = "GB" #filtering method (GB = gaussian blur, MA = moving average)
convSquare = 3

if __name__ == "__main__":
    for formula in metabolites:
        m0Mz,mzsOI,numCarbons = SIMSIToolBox.getMzsOfIsotopologues(formula,elementOfInterest="C")
        msi = SIMSIToolBox.MSIData(mzsOI,ppm=ppmThresh,numCores = num_cores,intensityCutoff=0)
        df = pd.read_csv(datadir + fn,index_col=0)
        msi.from_pandas(df,polarity)
        msi.smoothData(filt, convSquare)
        poolSize = np.sum(msi.data_tensor / msi.tic_image, axis=0)
        #poolSize = np.sum(msi.data_tensor, axis=0)

        plt.figure()
        SIMSIToolBox.showImage(poolSize, cmap=colormap)
        plt.title(formula)

        iso_tensor = SIMSIToolBox.normalizeTensor(msi.data_tensor)
        i = 0
        for image in iso_tensor:
            plt.figure()
            SIMSIToolBox.showImage(image, cmap=colormap)
            plt.title(formula + " M" + str(i))
            i += 1
            break

    plt.figure()
    SIMSIToolBox.showImage(msi.tic_image, cmap=colormap)

    plt.show()



