# Unsupervised constrastive learning of code embeddings

Final project for Stanford's CS329M (Machine Programming).

## Data

The data for this project comes from [Project CodeNet](https://developer.ibm.com/exchanges/data/all/project-codenet/). We provide a Python script <code>download_data.py</code> for downloading and expanding all of the data into the appropriate directories. To download all of the data used in this project, you can run:

```
python download_data.py --data_type cass
```

### Preprocessing

To convert the CASS data into DenseGraphs we can pass through our GNNs, run the following command:

```
python preprocess_cass.py
```

This script defaults to treating CASS nodes using the configuration used by [MISIM](https://arxiv.org/abs/2006.05265) (2-1-3-1-1). To learn more about other configurations, execute <code>python preprocess.py --help</code>.
