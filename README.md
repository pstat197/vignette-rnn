## PSTAT 197A Fall 2025 Table 1 Vignette Project

Vignette on building a bidirectional LSTM model for sentiment analysis using the IMDB movie review dataset; created as a class project for PSTAT 197A (Fall 2025).

# Contributors
- Quinlan Wilson
- Akhil Gorla
- Ella Yang
- Lorreta Lu
- Anish Senthil  

# Abstract
This vignette provides a practical introduction to Recurrent Neural Networks (RNNs), focusing on the implementation of a bidirectional Long Short-Term Memory (LSTM) model for text sentiment classification. Using the IMDB movie review dataset, we demonstrate preprocessing steps such as tokenization, label encoding, vocabulary restriction, sequence padding, and word embeddings. The vignette walks through model construction, training, evaluation, and visualization of learned word embeddings. By the end of the tutorial, readers will understand how LSTMs process sequential text data and how they can be applied to real-world NLP tasks.

# Repository Contents
This repository is organized as follows:
root directory
|-- data/
| |-- IMDB Dataset.csv # raw IMDB movie review dataset used for training and evaluation
| |-- archive.zip # original compressed dataset download
| |-- .DS_Store # auto-generated system file (not used)
|
|-- scripts/
| |-- LorrettaTest.R # exploratory R script for early testing
| |-- ModelDraftQuinlan.R # draft RNN/LSTM model implementation in R (Preferred)
| |-- ModelDraftAkhil.py # draft GRU sentiment model implemented in Python
|
|-- draft.ipynb # initial exploratory notebook for model investigations
|-- vignette-rnn.Rproj # RStudio project file for this vignette
|-- README.md # repository documentation
|-- .gitignore # files and folders excluded from version control

- **data/** contains the raw IMDB movie review dataset and any processed files used during modeling.  
- **scripts/** includes all code required to replicate model training, embedding extraction, PCA visualization, and predictions.  
- **vignette.qmd** is the primary document integrating narrative, equations, code, and outputs.  
- **vignette.html** is the rendered vignette for easy viewing.

# How to Use
To reproduce all results from the vignette:

1. Clone this repository.  
2. Open `scripts/ModelDraftQuinlan.R`.  
3. Run the script sequentially.  
This will preprocess the data, train the LSTM model, generate visualizations, and reproduce all results shown in the vignette.

# References
- S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," in Neural Computation, vol. 9, no. 8, pp. 1735-1780, 15 Nov. 1997, doi: 10.1162/neco.1997.9.8.1735.
- Rumelhart, D., Hinton, G. & Williams, R. Learning representations by back-propagating errors. Nature 323, 533–536 (1986). https://doi.org/10.1038/323533a0
- Maas, Andrew & Daly, Raymond & Pham, Peter & Huang, Dan & Ng, Andrew & Potts, Christopher. (2011). Learning Word Vectors for Sentiment Analysis. 142-150. 
