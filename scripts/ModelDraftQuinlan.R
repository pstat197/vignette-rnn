# Goal: Build a Recurrent Neural Network that can classify
#       whether a review is positive or negative based on
#       the test in said review
  
# Packages for RNN
library(keras)       # For deep learning
library(tidyverse)    # For read_csv and data manipulation
library(caret)        # For train/test split
library(tensorflow)
# Read in the data
movie_reviews <- read_csv("data/IMDB Dataset.csv")
head(movie_reviews)

set.seed(3434)

# Select the column with info on the emotion of review
sentiment <- movie_reviews$sentiment

# Check class balance (should be about equal,
# in our case exactly equal)
table(sentiment)

# Convert labels to binary (1 = positive, 0 = negative)
movie_reviews$sentiment_binary <- ifelse(movie_reviews$sentiment == "positive", 1, 0)

# tokenization (converting text into integers)
# Neural networks can't work with raw text so tokenization
# is used. On a basic level tokenization changes each word
# into an integer based on frequency. 

# We only use the top 10k most frequent words
max_words <- 10000

# RNNs need all sequences to be the same length.
# So by choosing 500 if a review has less than 500 words
# those extras are removed and if it has less than 500
# we add zeros to make it the same length. 
max_len <- 250
review_len <- sapply(strsplit(movie_reviews$review, " "), length)
hist(review_len, breaks = 50, main="Distribution of Review Lengths", xlab = "Words per review", xlim = c(0, 1250))
summary(review_len)

# Create a tokenizer and fit it to our review text
tokenizer <- text_tokenizer(num_words = max_words)
tokenizer %>% fit_text_tokenizer(movie_reviews$review)

word_counts <- tokenizer$word_counts
word_freq <- sort(unlist(word_counts), decreasing = T)
# Top 20 words
head(word_freq, 20)
# How many words appear only once or twice
sum(word_freq == 1)
sum(word_freq == 2)
# Since only 52k words appear once and 15k twice setting
# max words to 10k makes sense as these rare words add
# much to our classification model.

# Convert each review to a sequence of integers
sequences <- texts_to_sequences(tokenizer, movie_reviews$review)

# Make every sequences have the same length
x <- pad_sequences(sequences = sequences, maxlen = max_len)

y <- movie_reviews$sentiment_binary

# split the dataset into training and testing (80/20)
train_index <- createDataPartition(y, p = 0.8, list = F)

x_train <- x[train_index, ]
x_test <- x[-train_index, ]

y_train <- y[train_index]
y_test <- y[-train_index]

# Build the RNN model:
# The type of RNN used is called a Bidirectional LSTM
# (Long Short Term Memory) which means it takes into
# account the sequence of words in front and behind it.
# By doing this it helps our model infer context better 
# because the meaning of words can depend on the words
# that follow or proceed it.

# Embedding layer converts each work into a vector of
# length 64. To select the right embedding dimension you
# have to consider that larger values increase compute
# and risk overfitting while small values lose meaning. 
embedding_dim <- 64

model <- keras_model_sequential() %>%
  # Learn a 32-d vector for each word
  layer_embedding(
    input_dim = max_words,
    output_dim = embedding_dim,
    input_length = max_len
  ) %>%
  # Units controls how many neurons are in a layer where more
  # neurons mean the model can find my complex relationships 
  # although it can't be too large where overfitting poses a risk.
  # Overfitting is also why dropout is included. 
  bidirectional(layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2)) %>%
  # Output layer for binary classification
  layer_dense(units = 1, activation = "sigmoid")

# Layer 1: Embedding (250 words w/ 64 dimension vectors
#          per word)
#          Param #: represents the meaning of each word
#                   max_words(10k) * embedding_dim(64) = 640k
# Layer 2: Bidirectional LSTM 
#          (64 units in front + 64 behind = 128 total)
#          Param #: represents the learned weights inside the
#                   LSTM
# Layer 3: Dense (1 neuron to predict positive or negative)
#          Param #: represents the weights connecting 128 LSTM
#                   outputs to 1 output + 1 bias
# Total params: represents the total num of trainable and
#               not trainable params in the model
# Trainable params: are those which are updated during training
# Optimizer params: are internal params used by Adam to update
#                   the learning rates
summary(model)

# Compile model:
model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

# Train the model

# Fit for 3 epochs (passes over the entire dataset)
# validation split means 20% of training data is used
# to monitor overfitting
history <- model %>% fit(
  x_train, y_train,
  epochs = 3,
  batch_size = 30,
  validation_split = 0.2
)

# Plot accuracy and loss over epochs (passes)
plot(history)

# Visualizing the word embeddings in 2D via PCA (since we
# can't visualize 64 dimensions)

# First get weights from the first (embedding) layer
embeddings <- get_weights(model$layers[[1]])[[1]]

pca_res <- prcomp(embeddings[2:1000, ])
# Take the first 2 principal components
pca_df <- data.frame(
  PC1 = pca_res$x[,1],
  PC2 = pca_res$x[,2],
  word = names(tokenizer$word_index)[1:999]
)

ggplot(pca_df, aes(x = PC1, y = PC2, label=word)) +
  geom_text(size=3) +
  ggtitle("Word Embeddings PCA Projection")
# The graph shows the 2D projection of embeddings learned
# by the model. Where the position of the word shows how
# the model thinks that word relates to others. As you can
# see positive words like great, wonderful, favorite are
# together as well as negative words like awful, worst,
# terrible. Where negative and positive words are far apart
# and in the center there's a blob of less specific words. 


# Evaluate the model on the test data
model %>% evaluate(x_test, y_test)

examples <- c(
  "This is the best movie I have ever seen!",
  "The acting in this movie was horrible."
)
examples_seq <- texts_to_sequences(tokenizer, examples)
examples_pad <- pad_sequences(examples_seq, maxlen= max_len)
predictions <- model %>% predict(examples_pad)
pred_label <- ifelse(predictions > 0.5, "positive", "negative")
pred_label
