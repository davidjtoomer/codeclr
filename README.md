# Unsupervised learning of code embeddings using graph autoencoders
Final project for Stanford's CS329M (Machine Programming).

## Data

The data for this project comes from [Project CodeNet](https://developer.ibm.com/exchanges/data/all/project-codenet/). We provide a Python script <code>download_data.py</code> for downloading and expanding all of the data into the appropriate directories. To download all of the data used in this project, you can run:

```
python download_data.py --data_dir data --benchmark 1000 1400 --datatype code spts cass
```
