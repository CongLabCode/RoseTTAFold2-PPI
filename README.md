# RoseTTAFold2-PPI
A fast deep learning method for large-scale protein-protein interaction screening.

## Installation

1. Download the environment image from one of the following links:

   - [Download from conglab.swmed.edu](https://conglab.swmed.edu/downloads/SE3nv.sif)
   - [Download from prodata.swmed.edu](http://prodata.swmed.edu/humanPPI/bulk_download/SE3nv.sif)

2. Clone the repository:

   ```bash
   git clone https://github.com/CongLabCode/RoseTTAFold2-PPI.git

3. Download the weights to RoseTTAFold2-PPI/model:

   ```bash
   cd RoseTTAFold2-PPI/models
   wget https://conglab.swmed.edu/humanPPI/RF2-PPI.pt

## Usage
Running using singularity image:
```bash
singularity exec --bind /scratch/bcgc/jzhang21/human_PPI/stage10_RF2t_AF:/home/jzhang21 --nv ./SE3nv-20230612.sif /bin/bash -c "cd /home/jzhang21/;python RF2t_MONO_DDI_PPI/predict_list_PPI.py A list_adc"
