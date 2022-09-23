import SIMSIToolBox
import os
import matplotlib.pyplot as plt
import pandas as pd

inpath = "X:/MSI_Shared_Data/13CImagingManuscript/raw_data/imzmls/subset_data/"
outpath = "X:/MSI_Shared_Data/13CImagingManuscript/raw_data/imzmls/subset_data/NA_corrected_data/"

inpath = "X:/Kevin/Bruker/MALDI/imzML/processed_data/"
outpath = inpath + "NA_corrected/"



metabolites = ["C6H8NO5","C6H12O9P","C16H31O2","C20H31O2","C5H8NO4"]
ppmThresh = 20
num_cores = 20

polarity = "negative"

if __name__ == "__main__":
    files = [x for x in os.listdir(inpath) if ".csv" in x]
    for file in files:
        for met in metabolites:
            if file.replace(".csv","_"+met+".csv") not in os.listdir(outpath):
                m0Mz, mzsOI, numCarbons = SIMSIToolBox.getMzsOfIsotopologues(met, elementOfInterest="C")
                msi = SIMSIToolBox.MSIData(mzsOI,ppm=ppmThresh,numCores = num_cores,intensityCutoff=0)
                df = pd.read_csv(inpath+file,index_col=0)
                msi.from_pandas(df, "negative")

                msi.correctNaturalAbundance(met)

                df = msi.to_pandas()
                df.to_csv(outpath+file.replace(".csv","_"+met+".csv"))

