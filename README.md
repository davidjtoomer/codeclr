# Unsupervised constrastive learning of code embeddings
Final project for Stanford's CS329M (Machine Programming).

## Data

The data for this project comes from [Project CodeNet](https://developer.ibm.com/exchanges/data/all/project-codenet/). We provide a Python script <code>download_data.py</code> for downloading and expanding all of the data into the appropriate directories. To download all of the data used in this project, you can run:

```
python download_data.py --data_dir data --benchmark 1000 1400 --data_type code spts cass
```

### Preprocessing

To convert the CASS data into DenseGraphs we can pass through our GNNs, run the following command:

```
python preprocess_cass.py
```

This script default to treating CASS nodes as in simplified parse trees (SPT). If you want to use the configuration used by MISIM (2-1-3-1-1), you can specify the variables like so:

```
python preprocess_cass.py --annot_mode 2 --compound_mode 1 --gvar_mode 3 --gfun_mode 1 --fsig_mode 1
```

To learn more about the arguments you can pass to the function, execute <code>python preprocess_cass.py --help</code>.
