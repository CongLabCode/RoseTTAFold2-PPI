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
   wget --no-check-certificate https://conglab.swmed.edu/humanPPI/downloads/RF2-PPI.pt 

## Usage
To run RoseTTAFold2-PPI using the Singularity image, use the following command:

```bash
singularity exec \
  --bind /path/to/input_and_output_directory:/work/users \
  --bind /path/to/rosettafold2-ppi/directory:/home/RoseTTAFold2-PPI \
  --nv SE3nv.sif \
  /bin/bash -c "cd /work/users && python /home/RoseTTAFold2-PPI/src/predict_list_PPI.py input_file"
```

### Input File Format

For the *input_file*, e.g., examples/input_file, each line should contain two columns:

1. **File path** of the concatenated pairwise multiple sequence alignment (MSA) input.
2. **Length** of the first protein.

**Note**: When using Singularity, paths should be relative to the directories mounted inside the container. If you prefer to use absolute paths, ensure they reference the file paths **inside the container** after mounting the directories.

### Output File
The output file will be saved as `[input_filename].npz`, where `input_filename` is the name of your input file.


### Test
```bash
cd RoseTTAFold2-PPI
exec_dir=$(pwd)
singularity exec \
    --bind $exec_dir:/home/RoseTTAFold2-PPI \
    --nv SE3nv.sif \
    /bin/bash -c "cd /home/RoseTTAFold2-PPI && python /home/RoseTTAFold2-PPI/src/predict_list_PPI.py examples/test.list"
```

The command will generate `test.list.log` and `test.list.npz` under `RoseTTAFold2-PPI/examples` which should be the same as files under `examples/expected_output`.
