# RoseTTAFold2-PPI
A fast deep learning method for large-scale protein-protein interaction screening.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/CongLabCode/RoseTTAFold2-PPI.git
   ```

2. Download the weights to RoseTTAFold2-PPI/src/model:

   ```bash
   cd RoseTTAFold2-PPI/src/models
   wget --no-check-certificate https://conglab.swmed.edu/humanPPI/downloads/RF2-PPI.pt
   ```

3. Choose one of the following installation methods:

   A. Download our singularity image (hhsuite is needed if you want to use our paired MSA generation script):
      
      ```bash
      cd RoseTTAFold2-PPI/
      wget --no-check-certificate https://conglab.swmed.edu/humanPPI/downloads/SE3nv-20230612.sif
      conda install -c conda-forge -c bioconda hhsuite
      ```

   B. Install conda environment (if cannot use singularity):
      
      ```bash
      conda create -n rf2ppi python=3.9
      conda activate rf2ppi
      conda install -c conda-forge -c bioconda hhsuite
      pip install numpy==1.21.2
      pip install pandas==1.5.3
      pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
      pip install biopython==1.79
      pip install scipy==1.7.1
      pip install einops
      ```

<br>

## Usage

1. using singularity:

   ```bash
   singularity exec \
     --bind /path/to/input_and_output_directory:/work/users \
     --bind /path/to/rosettafold2-ppi/directory:/home/RoseTTAFold2-PPI \
     --nv SE3nv-20230612.sif \
     /bin/bash -c "cd /work/users && python /home/RoseTTAFold2-PPI/src/predict_list_PPI.py -list_fn input_file -model_file model_file"
   ```

2. using conda environment:

   ```bash
   conda activate rf2ppi
   python [/path/to/]RoseTTAFold2-PPI/src/predict_list_PPI.py -list_fn [input_file] -model_file [/path/to]/RoseTTAFold2-PPI/src/models/RF2-PPI.pt

<br>

## Input

### Building your own paired MSAs

For the *[input_file]*, e.g., examples/input_file, each line should contain two columns:

1. **File path** of the paired multiple sequence alignment (MSA) input.
2. **Length** of the first protein.

**Note**: When using Singularity, paths should be relative to the directories mounted inside the container. If you prefer to use absolute paths, ensure they reference the file paths **inside the container** after mounting the directories.

A simplified pipeline to generate paired MSAs is as follows:

1. Search homologs for each protein using tools like HHblits (be sure to turn on "-all" flag in HHblits to output all the hits)
2. Identify the closest hit (by sequence identity) to the query from each organism, and discard other hits
3. Combine the resulting MSAs of both proteins by concatenating the two hit sequences (one for each query) from the same organism
4. Discard sequences that cannot be paired
5. Remove redundancy by 90% or 95% sequence identity using hhfilter.

### Using omicMSA

For most human proteins, you can generate paired MSAs using the omicMSAs for single proteins we shared at https://conglab.swmed.edu/humanPPI/humanPPI_download.html

We share omicMSAs both as entire proteins (protein_omicMSAs.tar.gz) and segments (segment_omicMSAs.tar.gz, breaking long proteins into shorter segments and excluding low-quality positions in the MSAs).

### Commands for generating paired MSAs from omicMSAs

To generate paired MSAs from the omicMSAs we shared, please use the following commands:

#### For proteins:

```bash
python /path/to/RoseTTAFold2-PPI/generate_protein_pair_MSA.py [list_of_protein_pairs] [directory_with_single_protein_MSAs] [output_directory_with_paired_MSAs]
```

#### For segments:

```bash
python /path/to/RoseTTAFold2-PPI/generate_segment_pair_MSA.py [list_of_segment_pairs] [directory_with_single_segment_MSAs] [output_directory_with_paired_MSAs]
```

### Best practices

**Note:** The performance is affected by the quality of the paired MSAs. Our benchmarks suggest the following best practice can enhance the accuracy of RoseTTAFold2-PPI:

1. Get deeper MSAs
2. Remove low-quality regions, corresponding to poorly conserved intrinsically disordered regions
3. Only include paired MSAs and remove any unpaired sequences
4. Remove redundancy sequences at 90% or 95% sequence identity after "pairing"

<br>

## Output

The output file will be saved as `[input_file].npz` and `[input_file].log`, e.g., those in `examples/expected_output`.

### Log File Format (`[input_file].log`)

The log file contains three columns:

1. The input MSA file name
2. Predicted Interaction probability for a protein/segment pair
3. Compute time

### NPZ File Format (`[input_file].npz`)

The npz file contains the inter-residue interaction probabilities. The input MSA file names were used as keys that point to a numpy matrix containing the predicted interaction probability between a residue in the first protein and a residue in the second. This matrix has the shape of `(L1, L2)`, where `L1` and `L2` are the lengths of the two proteins.

### Important Note on Prediction Variability

Similar to AlphaFold2, predicted interaction probability by RoseTTAFold-PPI is not deterministic. Our benchmark suggests that the standard deviation in predicted interaction probabilities might exceed 0.1 (out of 1) for about 5% of cases (**Fig. A** below), and such variability is more obvious for pairs with intermediate interaction probabilities (**Fig. B** below).

![alt text](https://github.com/CongLabCode/RoseTTAFold2-PPI/blob/main/rf2_ppi.png?raw=true)

<br>

## Test

1. using singularity:

   ```bash
   cd RoseTTAFold2-PPI
   exec_dir=$(pwd)
   singularity exec \
       --bind $exec_dir:/home/RoseTTAFold2-PPI \
       --nv SE3nv.sif \
       /bin/bash -c "cd /home/RoseTTAFold2-PPI && python /home/RoseTTAFold2-PPI/src/predict_list_PPI.py -list_fn examples/protein_pairs_input -model_file src/models/RF2-PPI.pt"
   ```
2. using conda environment:
   ```bash
   conda activate rf2ppi
   cd RoseTTAFold2-PPI/examples
   python ../src/predict_list_PPI.py -list_fn segment_pairs_input -model_file ../src/models/RF2-PPI.pt
   ```

These commands will generate outputs similar to those in examples/expected_output.

<br>

## Reference

Jing Zhang*, Ian R Humphreys*, Jimin Pei*, Jinuk Kim, Chulwon Choi, Rongqing Yuan, Jesse Durham, Siqi Liu, Hee-Jung Choi, Minkyung Baek, David Baker, Qian Cong. **Computing the Human Interactome.** (https://www.biorxiv.org/content/10.1101/2024.10.01.615885v1)
