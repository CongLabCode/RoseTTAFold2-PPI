# RoseTTAFold2-PPI
A fast deep learning method for large-scale protein-protein interaction screening.

## Installation

1. Download the environment image from one of the following links:

   - [Download from conglab.swmed.edu](https://conglab.swmed.edu/humanPPI/SE3nv.sif)
   - [Download from prodata.swmed.edu](http://prodata.swmed.edu/humanPPI/bulk_download/SE3nv.sif)

2. Clone the repository:

   ```bash
   git clone https://github.com/CongLabCode/RoseTTAFold2-PPI.git

3. Download the weights to RoseTTAFold2-PPI/src/model:

   ```bash
   cd RoseTTAFold2-PPI/src/models
   wget https://conglab.swmed.edu/humanPPI/RF2-PPI.pt

## Usage
To run RoseTTAFold2-PPI using the Singularity image, use the following command:

```bash
singularity exec \
  --bind /path/to/output_directory:/work/users \
  --bind /path/to/rosettafold2-ppi/directory:/home/RoseTTAFold2-PPI \
  --nv SE3nv-20230612.sif \
  /bin/bash -c "cd /work/users; python /home/RoseTTAFold2-PPI/src/predict_list_PPI.py input_file"
```

### Input File Format

For the *input_file*, e.g., examples/input_file, each line should contain three columns:

1. **File path** of the multiple sequence alignment (MSA) input.
2. **Length** of the first protein.
3. **File path** for the output.

