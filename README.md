# Chai-2022-GCaMP
Code used for calcium data analysis in "Interneuron Control of C. elegans Developmental Decision-making".  

### Data Acquisition
For data acquisition, a customized confocal system capable of performing fast volumeric imaging is used.
For more information about the system visit: [Lambda github](https://github.com/venkatachalamlab/lambda)

### Manual Annotation
To annotate neurons in recordings, a costomized software is used. This software uses a web based user interface, and visualizes datasets to make manual annotations easier. The result is saved as a Pandas dataframe in a hdf file. For more information about the software visit: [annotator github](https://github.com/venkatachalamlab/annotator)

### Preprocessing
To prepare datasets for trace extraction, first we used the 75 percent quantile as threshold to subtract background noise, and converted data type to 'uint8'. Next, we used annotator (see above) to label all neurons at all timepoints. We used the coordiantes of neurons to extract traces. For trace extraction, we use a fixed number of pixels in each neuron that have the highest intensity. In case there is another neuron close to the neuron of interest, we exclude pixels that are closer to the neighboring neurons than the neuron of interest.

### Plotting
Traces from the preprocessing step, and the times of stimulus delivery (saved as a txt file for each dataset) are used to make plots.
