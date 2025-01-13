#!/usr/bin/env python
# coding: utf-8

# Version: 02.14.2023

# # Lab 6.2: Implementing Topic Extraction with NTM
# 
# In this lab, you will use the Amazon SageMaker Neural Topic Model (NTM) algorithm to extract topics from the [20 Newsgroups](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups) dataset.
# 
# The Amazon SageMaker Neural Topic Model (NTM) is an unsupervised learning algorithm that attempts to describe a set of observations as a mixture of distinct categories. NTM is most commonly used to discover a user-specified number of topics that are shared by documents within a text corpus.
# 
# Each observation is a document, the features are the presence (or occurrence count) of each word, and the categories are the topics. Because the method is unsupervised, the topics are not specified up front and are not guaranteed to align with how a human might naturally categorize documents. The topics are learned as a probability distribution over the words that occur in each document. Each document, in turn, is described as a mixture of topics. For more information, see [Neural Topic Model (NTM) Algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/ntm.html) in the Amazon SageMaker Developer Guide.
# 
# 
# ## About this dataset
# 
# The 20 Newsgroups dataset is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. This collection has become a popular dataset for experiments in text applications of machine learning techniques, such as text classification and text clustering. In this lab, you will see what topics you can learn from this set of documents using the Neural Topic Model (NTM) algorithm.
# 
# Dataset source: Tom Mitchell. *20 Newsgroups Data*. September 9, 1999. Distributed by UCI KDD Archive. https://kdd.ics.uci.edu/databases/20newsgroups/20newsgroups.data.html.

# ## Lab steps
# 
# 1. [Fetching the dataset](#1.-Fetching-the-dataset)
# 2. [Examining and preprocessing the data](#2.-Examining-and-preprocessing-the-data)
# 3. [Preparing the data for training](#3.-Preparing-the-data-for-training)
# 4. [Training the model](#4.-Training-the-model)
# 5. [Using the model for inference](#5.-Using-the-model-for-inference)
# 6. [Exploring the model](#6.-Exploring-the-model)
# 
# ## Submitting your work
# 
# 1. In the lab console, choose **Submit** to record your progress and when prompted, choose **Yes**.
# 
# 1. If the results don't display after a couple of minutes, return to the top of the lab instructions and choose **Grades**.
# 
#     **Tip:** You can submit your work multiple times. After you change your work, choose **Submit** again. Your last submission is what will be recorded for this lab.
# 
# 1. To find detailed feedback on your work, choose **Details** followed by **View Submission Report**.

# ## 1. Fetching the dataset
# ([Go to top](#Lab-6.2:-Implementing-Topic-Extraction-with-NTM))
# 
# First, define the folder to hold the data. Then, clean up the folder, which might contain data from previous experiments.

# In[1]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install --upgrade SageMaker')
get_ipython().system('pip install --upgrade nltk')


# In[2]:


import boto3
import os
import shutil

def check_create_dir(dir):
    if os.path.exists(dir):  # Clean up existing data folder
        shutil.rmtree(dir)
    os.mkdir(dir)

data_dir = '20_newsgroups'
check_create_dir(data_dir)


# In the next two cells, you unpack the dataset and extract a list of the files.

# In[3]:


get_ipython().system('tar -xzf ../s3/20_newsgroups.tar.gz')
get_ipython().system('ls 20_newsgroups')


# In[4]:


folders = [os.path.join(data_dir,f) for f in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, f))]
file_list = [os.path.join(d,f) for d in folders for f in os.listdir(d)]
print('Number of documents:', len(file_list))


# ## 2. Examining and preprocessing the data
# ([Go to top](#Lab-6.2:-Implementing-Topic-Extraction-with-NTM))
#     
# In this section, you will examine the data and perform some standard natural language processing (NLP) data cleaning tasks.

# Remind yourself what the files look like in order to determine the best preprocessing steps to take.

# In[5]:


get_ipython().system('cat 20_newsgroups/comp.graphics/37917')


# Each newsgroup document can have the following sections:
# - header - Contains the standard newsgroup header information. This should be removed.
# - quoted text - Text from a previous message, which usually is prefixed with '>' or '|', and sometimes starts with *writes*, *wrote*, *said*, or *says*.
# - message - Body of the message that you want to extract topics from.
# - footer - Messages typically end with a signature.

# Define the following functions, which you will use to remove the headers, quoted text, and footers.

# In[6]:


import re
def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    """
    _before, _blankline, after = text.partition('\n\n')
    return after

_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')


def strip_newsgroup_quoting(text):
    """
    Given text in "news" format, strip lines beginning with the quote
    characters > or |, plus lines that often introduce a quoted section
    (for example, because they contain the string 'writes:'.)
    """
    good_lines = [line for line in text.split('\n')
                  if not _QUOTE_RE.search(line)]
    return '\n'.join(good_lines)


def strip_newsgroup_footer(text):
    """
    Given text in "news" format, attempt to remove a signature block.

    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).
    """
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text


# Import the packages you need for preprocessing the dataset.

# In[7]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from nltk.stem.wordnet import WordNetLemmatizer


# Next, remove extra spaces, convert the text to lowercase, and lemmatize the text.

# In[8]:


stop = stopwords.words('english')
lem = WordNetLemmatizer()

def clean(sent):
    # Implement this function
    sent = sent.lower()
    sent = re.sub('\s+', ' ', sent)
    sent = sent.strip()
    sent = re.compile('<.*?>').sub('',sent)
    # Remove special characters and digits
    sent=re.sub("(\\d|\\W)+"," ",sent)
    sent=re.sub("br","",sent)
    filtered_sentence = []
    
    for w in word_tokenize(sent):
        # You are applying custom filtering here. Feel free to try different things.
        # Check if it is not numeric, the length > 2, and it is not in stopwords.
        if(not w.isnumeric()) and (len(w)>2) and (w not in stop):  
            # Stem and add to filtered list
            filtered_sentence.append(lem.lemmatize(w))
    final_string = " ".join(filtered_sentence) # Final string of cleaned words
    return final_string


# Read each of the newsgroups messages. Remove the header, quotes, and footers. Then, store the results in an array.

# In[10]:


import nltk
nltk.download('punkt_tab')
data = []
source_group = []
for f in file_list:
    with open(f, 'rb') as fin:
        content = fin.read().decode('latin1')   
        content = strip_newsgroup_header(content)
        content = strip_newsgroup_quoting(content)
        content = strip_newsgroup_footer(content)
        content = clean(content)
        # Remove header, quoting, and footer
        data.append(content)
        


# As you can see, the entries in the dataset are now just plain text paragraphs. You need to process them into a data format that the NTM algorithm can understand.

# In[11]:


data[10:13]


# The next step is to vectorize the data so that it is ready for training. You can use `CountVectorizer`, which you have used in previous labs, and limit the vocabulary size to `vocab_size`. 
# 
# Use a maximum document frequency of 95 percent of documents (`max_df=0.95`) and a minimum document frequency of 2 documents (`min_df=2`).

# In[12]:


get_ipython().run_cell_magic('time', '', "import numpy as np\nfrom sklearn.feature_extraction.text import CountVectorizer\nvocab_size = 2000\nprint('Tokenizing and counting, this may take a few minutes...')\n\n# vectorizer = CountVectorizer(input='content', max_features=vocab_size, max_df=0.95, min_df=2)\nvectorizer = CountVectorizer(input='content', max_features=vocab_size)\nvectors = vectorizer.fit_transform(data)\nvocab_list = vectorizer.get_feature_names_out()\n\nprint('vocab size:', len(vocab_list))\n")


# Optionally, consider removing short documents. A short document is not likely to express more than one topic. Topic modeling tries to model each document as a mixture of multiple topics; therefore, topic modeling might not be suitable for short documents.
# 
# The following cell removes documents that contain fewer than 25 words.

# In[13]:


threshold = 25
vectors = vectors[np.array(vectors.sum(axis=1)>threshold).reshape(-1,)]
print('removed short docs (<{} words)'.format(threshold))        
print(vectors.shape)


# The output from `CountVectorizer` are sparse matrices with their elements being integers. 

# In[14]:


print(type(vectors), vectors.dtype)
print(vectors[0])


# All of the parameters (weights and biases) in the NTM model are `np.float32` type. Therefore, you need the input data to also be `np.float32` type. It is better to do this type-casting up front rather than repeatedly casting during mini-batch training.

# In[15]:


import scipy.sparse as sparse
vectors = sparse.csr_matrix(vectors, dtype=np.float32)
print(type(vectors), vectors.dtype)


# # 3. Preparing the data for training
# ([Go to top](#Lab-6.2:-Implementing-Topic-Extraction-with-NTM))
# 

# As a common practice in model training, you should have a training set, validation set, and test set. The training set is the set of data that the model is actually being trained on. You care about the model's performance on future, unseen data. Therefore, during training, you periodically calculate scores (or losses) on the validation set to validate the performance of the model on unseen data. By assessing the model's ability to generalize, you can stop training at the optimal point to avoid overtraining.
# 
# Note that when you only have a training set and no validation set, the NTM model will rely on scores on the training set to perform early stopping, which could result in overtraining. Therefore, you should always supply a validation set to the model.
# 
# In this lab, you will use 80 percent of the dataset as the training set, and the remaining 20 percent for the validation set and test set. You will use the validation set in training and use the test set to demonstrate model inference.

# In[16]:


from sklearn.model_selection import train_test_split
def split_data(df):
    train, test_validate = train_test_split(df,
                                            test_size=0.2,
                                            shuffle=True,
                                            random_state=324
                                            )
    test, validate = train_test_split(test_validate,
                                            test_size=0.5,
                                            shuffle=True,
                                            random_state=324
                                            )
    return train, validate, test


# In[17]:


train_vectors, val_vectors, test_vectors = split_data(vectors)


# In[18]:


print(train_vectors.shape, val_vectors.shape)


# ## Save the vocabulary file
# 
# To make use of the auxiliary channel for the vocabulary file, first save the text file with the name **vocab.txt** in the **auxiliary** directory.
# 

# In[19]:


import os
import shutil
aux_data_dir = os.path.join(data_dir, 'auxiliary')
check_create_dir(aux_data_dir)
with open(os.path.join(aux_data_dir, 'vocab.txt'), 'w', encoding='utf-8') as f:
    for item in vocab_list:
        f.write(item+'\n')


# EDITOR COMMENTS for the following cell:
# - In the first sentence, are "recordIO" and "protobuf" two different formats, or are they one format? It's written as if they are one format, but the words link to different URLs. From [this page in the SageMaker documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html#cdf-recordio-format), I believe it should be written "protobuf recordIO format", but I don't understand why there are two URLs.
# - The URL for "RecordIO" gives a 404 error. The following may be a replacement URL: https://mxnet.apache.org/versions/1.8.0/api/python/docs/api/mxnet/recordio/index.html#mxnet-recordio.
# - The cell says "You will convert..." but then it looks like the actual conversion occurs several cells later. Consider relocating this cell to be closer to the code cell to which it applies.

# ## Store the data on Amazon S3
# 
# The NTM algorithm accepts data in the [recordIO - protobuf](https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html) format. The SageMaker Python API provides helper functions to convert your data into this format. You will convert the data from NumPy/SciPy and then upload it to an Amazon Simple Storage Service (Amazon S3) destination for the model to access during training.
# 

# EDITOR COMMENT for the following cell: The cell references "boto regexp", but I don't see that text in any of the code cells within this lab. Should "boto regexp" be in one of the code cells? Should "boto regexp" be replaced with something else here? Is that note applicable to this lab?

# ## Set up AWS credentials
# 
# You first need to specify data locations and access roles. In particular, you need the following data:
# 
# - The S3 `bucket` and `prefix` that you want to use for the training and model data. This should be within the same Region as the notebook instance, training, and hosting.
# - The AWS Identity and Access Management (IAM) `role` is used to give training and hosting access to your data. See the documentation for how to create these. **Note:** If more than one role is required for notebook instances, training, and/or hosting, replace the `boto regexp` with the appropriate full IAM role Amazon Resource Number (ARN) string or strings.
# 
# **Note:** These values will have been supplied when the lab starts.

# In[20]:


import os
import sagemaker
from sagemaker import get_execution_role

role = get_execution_role()

sess = sagemaker.Session()
bucket = "c137242a3503189l8673571t1w874328056043-labbucket-r48acfgty5d7"


# In[21]:


prefix = '20newsgroups-ntm'

train_prefix = os.path.join(prefix, 'train')
val_prefix = os.path.join(prefix, 'val')
aux_prefix = os.path.join(prefix, 'auxiliary')
output_prefix = os.path.join(prefix, 'output')

s3_train_data = os.path.join('s3://', bucket, train_prefix)
s3_val_data = os.path.join('s3://', bucket, val_prefix)
s3_aux_data = os.path.join('s3://', bucket, aux_prefix)
output_path = os.path.join('s3://', bucket, output_prefix)
print('Training set location', s3_train_data)
print('Validation set location', s3_val_data)
print('Auxiliary data location', s3_aux_data)
print('Trained model will be saved at', output_path)


# Now, define a helper function to convert the data to the recordIO protobuf format and upload it to Amazon S3. In addition, you will have the option to split the data into several parts as specified by `n_parts`.
# 
# The algorithm inherently supports multiple files in the training folder ("channel"), which could be helpful for a large dataset. In addition, when you use distributed training with multiple workers (compute instances), having multiple files enables you to distribute different portions of the training data to different workers.
# 
# This helper function uses the `write_spmatrix_to_sparse_tensor` function, provided by the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk), to convert SciPy sparse matrix into the recordIO protobuf format.

# In[22]:


def split_convert_upload(sparray, bucket, prefix, fname_template='data_part{}.pbr', n_parts=2):
    import io
    import boto3
    import sagemaker.amazon.common as smac
    
    chunk_size = sparray.shape[0]// n_parts
    for i in range(n_parts):

        # Calculate start and end indices
        start = i*chunk_size
        end = (i+1)*chunk_size
        if i+1 == n_parts:
            end = sparray.shape[0]
        
        # Convert to record protobuf
        buf = io.BytesIO()
        smac.write_spmatrix_to_sparse_tensor(array=sparray[start:end], file=buf, labels=None)
        buf.seek(0)
        
        # Upload to S3 location specified by bucket and prefix
        fname = os.path.join(prefix, fname_template.format(i))
        boto3.resource('s3').Bucket(bucket).Object(fname).upload_fileobj(buf)
        print('Uploaded data to s3://{}'.format(os.path.join(bucket, fname)))


# In[23]:


split_convert_upload(train_vectors, bucket=bucket, prefix=train_prefix, fname_template='train_part{}.pbr', n_parts=8)
split_convert_upload(val_vectors, bucket=bucket, prefix=val_prefix, fname_template='val_part{}.pbr', n_parts=1)


# Upload the vocab.txt file.

# In[24]:


boto3.resource('s3').Bucket(bucket).Object(aux_prefix+'/vocab.txt').upload_file(aux_data_dir+'/vocab.txt')


# # 4. Training the model
# ([Go to top](#Lab-6.2:-Implementing-Topic-Extraction-with-NTM))
# 
# You have created the training and validation datasets and uploaded them to Amazon S3. Next, configure a SageMaker training job to use the NTM algorithm on the data that you prepared.

# In[25]:


from sagemaker.image_uris import retrieve
container = retrieve('ntm',boto3.Session().region_name)


# The code in the following cell automatically chooses an algorithm container based on the current Region. In the API call to `sagemaker.estimator.Estimator`, you also specify the type and count of instances for the training job. Because the 20 Newsgroups dataset is relatively small, you can use a CPU-only instance (`ml.c4.xlarge`).
# 
# NTM fully takes advantage of GPU hardware and, in general, trains roughly an order of magnitude faster on a GPU than on a CPU. Multi-GPU or multi-instance training further improves training speed roughly linearly if communication overhead is low compared to compute time.

# In[26]:


import sagemaker
sess = sagemaker.Session()
ntm = sagemaker.estimator.Estimator(container,
                                    role, 
                                    instance_count=2, 
                                    instance_type='ml.c4.xlarge',
                                    output_path=output_path,
                                    sagemaker_session=sagemaker.Session())


# EDITOR COMMENT for following cell: This appears to be the only subsection within this task. In general, don't use subsections if only one exists. Recommend removing this heading or adding headings to other subsections within this task.

# ## Set the hyperparameters
# 
# The following is a partial list of hyperparameters. For the full list of available hyperparameters, see [NTM Hyperparameters](https://docs.aws.amazon.com/sagemaker/latest/dg/ntm_hyperparameters.html) in the Amazon SageMaker Developer Guide.
# 
# - **feature_dim** - The "feature dimension", which should be set to the vocabulary size
# - **num_topics** - The number of topics to extract
# - **mini_batch_size** - The batch size for each worker instance. Note that in multi-GPU instances, this number will be further divided by the number of GPUs. For example, if you plan to train on an 8-GPU machine (such as `ml.p2.8xlarge`) and want each GPU to have 1024 training examples per batch, `mini_batch_size` should be set to 8196.
# - **epochs** - The maximum number of epochs to train for; training may stop early
# - **num_patience_epochs** and **tolerance** - Control the early stopping behavior. In general, early stopping occurs when there hasn't been improvement on validation loss within the last `num_patience_epochs` number of epochs. Improvements smaller than `tolerance` are considered non-improvement.
# - **optimizer** and **learning_rate** - The default optimizer is `adadelta`, and `learning_rate` does not need to be set. For other optimizers, the choice of an appropriate learning rate may require experimentation.

# In[27]:


num_topics = 20
ntm.set_hyperparameters(num_topics=num_topics, 
                        feature_dim=vocab_size, 
                        mini_batch_size=256, 
                        num_patience_epochs=10, 
                        optimizer='adam')


# Next, specify how the training data and validation data will be distributed to the workers during training. Data channels have two modes:
# 
# - `FullyReplicated`: All data files will be copied to all workers.
# - `ShardedByS3Key`: Data files will be sharded to different workers. Each worker will receive a different portion of the full dataset.
# 
# The Python SDK uses the `FullyReplicated` mode for all data channels by default. This is desirable for the validation (test) channel but not for training channel. The reason is that when you use multiple workers, you would like to go through the full dataset by having each worker go through a different portion of the dataset to provide different gradients within epochs. When you use the `FullyReplicated` mode on training data, the training time per epoch is slower (nearly 1.5 times in this example), and it defeats the purpose of distributed training. To set the training data channel correctly, you specify `distribution` to be `ShardedByS3Key` for the training data channel.

# In[28]:


from sagemaker.inputs import TrainingInput
# sagemaker.inputs.TrainingInput
s3_train = TrainingInput(s3_train_data, distribution='ShardedByS3Key') 
s3_val = TrainingInput(s3_val_data, distribution='FullyReplicated')


# The final step before training is to define the auxiliary file. This will replace integers in the log files with the actual words.

# In[29]:


s3_aux = TrainingInput(s3_aux_data, distribution='FullyReplicated', content_type='text/plain')


# Now you are ready to train. The following cell takes a few minutes to run. The command will first provision the required hardware. You will see a series of dots indicating the progress of the hardware provisioning process. Once the resources are allocated, training logs will be displayed. With multiple workers, the log color and the ID following `INFO` identifies logs that are emitted by different workers.

# In[30]:


# ntm.fit({'train': s3_train, 'validation': s3_train, 'auxiliary': s3_aux})
ntm.fit({'train': s3_train, 'validation': s3_val, 'auxiliary': s3_aux})


# If you see the message
# 
# > `===== Job Complete =====`
# 
# at the bottom of the output logs, then training has successfully completed, and the output NTM model was stored in the specified output path.
# 
# You can also view information about a training job on the SageMaker console. In the left navigation pane, under **Training**, choose **Training jobs**. Then, select the training job that matches the training job name from the following cell's output.

# In[31]:


print('Training job name: {}'.format(ntm.latest_training_job.job_name))


# In the cell above that contains the log information for the training job, scroll until you find a line similar to the one in the following cell.
# 
# **Tip:** Look for the phrase `Topics from epoch:final`.

#     [05/04/2021 02:01:05 INFO 140593644394304] Topics from epoch:final (num_topics:20) [wetc 0.33, tu 0.68]

# Two numbers are of interest here: **wetc** and **tu**.
# 
# - **wetc** is the *word embedding topic coherence* and indicates the degree of topic coherence. A higher number indicates a higher degree of topic coherence.
# - **tu** is the *topic uniqueness* metric and indicates how unique the terms are within the topic. A higher number indicates that the topic terms are more unique.
# 
# In the example cell, the wetc is average at 0.33, and the tu is above average at 0.68.

# After the line that displays the overall wetc and tu metrics, you should see a list of topics that were identified along with the words that comprise that topic. Note that the topics are not named. That task still requires a human. For each topic, you see its wetc and tu scores, as well as the top words within that topic. 
# 
# Review these words, and try to determine a name for each topic.
# 
# **Note:** Your results may be different than those in the following cell.

#     [05/04/2021 02:01:05 INFO 140593644394304] [0.60, 0.80] game win playoff player team espn season play detroit baseball cup league score pitcher nhl goal toronto played hockey montreal
# 
# Topic 0 seems to be about sports.
# 
#     [05/04/2021 02:01:05 INFO 140593644394304] [0.57, 1.00] christ jesus god scripture doctrine christian sin bible faith christianity atheist church religion islam holy heaven biblical eternal morality belief
# 
# Topic 1 seems to be about religion.
# 
#     [05/04/2021 02:01:05 INFO 140593644394304] [0.30, 0.97] scsi ide motherboard controller drive mhz connector isa slot bus cpu jumper pin meg adapter floppy simms external cache speed
#     
# Topic 2 seems to be about computers.
# 
# The following topic has a low uniqueness score. It's not clear what this topic would be. Is it something to do with motorcycles, food, or something else? You could name this '**unknown**'.
# 
#     [05/04/2021 02:01:05 INFO 140593644394304] [0.35, 0.39] clutch eat bike doctor stopped msg riding watching pitch wheel ride hit food pain feeling personally fix sometimes rear tend
#     
# 

# # 5. Using the model for inference
# ([Go to top](#Lab-6.2:-Implementing-Topic-Extraction-with-NTM))
# 
# Now that you have a trained NTM model, use it to perform inference on data. For this example, that means predicting the topic mixture that represents a given document.
# 
# To create an inference endpoint, use the SageMaker Python SDK `deploy()` function from the job that you defined previously. Specify the instance type where inference is computed as well as an initial number of instances to launch.

# In[32]:


ntm_predictor = ntm.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


# Congratulations! You now have a functioning SageMaker NTM inference endpoint.
# 
# You can confirm the endpoint configuration and status in the SageMaker console. In the left navigation pane, under **Inference**, choose **Endpoints**. Then, select the endpoint that matches the endpoint name from the following cell's output.

# In[33]:


print('Endpoint name: {}'.format(ntm_predictor.endpoint_name))


# 
# ### Data serialization and deserialization
# 
# You can pass data in a variety of formats to the inference endpoint. First, you will pass CSV-formatted data. Use the SageMaker Python SDK utilities `csv_serializer` and `json_deserializer` to configure the inference endpoint.

# In[34]:


ntm_predictor.content_types = 'text/csv'
ntm_predictor.serializer = sagemaker.serializers.CSVSerializer()
ntm_predictor.deserializer = sagemaker.deserializers.JSONDeserializer()


# Now,  pass five examples from the test set to the inference endpoint.

# In[35]:


test_data = np.array(test_vectors.todense())
results = ntm_predictor.predict(test_data[:5])
print(results)


# The output format of the SageMaker NTM inference endpoint is a Python dictionary with the following format.
# 
# ```
# {
#   'predictions': [
#     {'topic_weights': [ ... ] },
#     {'topic_weights': [ ... ] },
#     {'topic_weights': [ ... ] },
#     ...
#   ]
# }
# ```
# 
# Extract the topic weights that correspond to each of the input documents.

# In[36]:


predictions = np.array([prediction['topic_weights'] for prediction in results['predictions']])
print(predictions)


# Replace the topic names in the following cell with the topic names that you determined.

# In[37]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
colnames = pd.DataFrame({'topics':['topic 0', 'topic 1', 'topic 2', 'topic 3', 'topic 4', 'topic 5', 'topic 6','topic 7','topic 8','topic 9',
       'topic 10', 'topic 11', 'topic 12', 'topic 13', 'topic 14', 'topic 15', 'topic 16','topic 17','topic 18','topic 19']})


# Now, use a bar plot to take a look at how the 20 topics are assigned to the 5 test documents.

# In[38]:


fs = 12
df=pd.DataFrame(predictions.T)
df.index = colnames['topics']
df.plot(kind='bar', figsize=(16,4), fontsize=fs)
plt.ylabel('Topic assignment', fontsize=fs+2)
plt.xlabel('Topic ID', fontsize=fs+2)


# You could improve the model by adding or removing specific words to influence topics, increasing or decreasing the number of topics, and trying different hyperparameters.

# ## Delete the endpoint
# 
# Finally, delete the endpoint before you close the notebook.
# 
# To restart the endpoint, you can follow the code in section 5 using the same `endpoint_name`.

# In[39]:


sagemaker.Session().delete_endpoint(ntm_predictor.endpoint_name)


# # 6. Exploring the model
# ([Go to top](#Lab-6.2:-Implementing-Topic-Extraction-with-NTM))

# **Note: This section provides a deeper exploration of the trained models. The demonstrated functionalities may not be fully supported or guaranteed. For example, the parameter names may change without notice.**
# 
# The trained model artifact is a compressed package of MXNet models from the two workers. To explore the model, you first need to install MXNet.

# In[50]:


# If you use the conda_mxnet_p36 kernel, MXNet is already installed; otherwise, uncomment the following line to install it.
#!pip install numpy==1.19.5
get_ipython().system('pip install mxnet --upgrade')
import numpy as np
np.bool = np.bool_#very very very imp....these 2 lines
import mxnet as mx


# In[ ]:


Download and unpack the artifact.


# In[51]:


model_path = os.path.join(output_prefix, ntm._current_job_name, 'output/model.tar.gz')
model_path


# In[52]:


boto3.resource('s3').Bucket(bucket).download_file(model_path, 'downloaded_model.tar.gz')


# In[53]:


get_ipython().system("tar -xzvf 'downloaded_model.tar.gz'")


# In[54]:


# Use flag -o to overwrite the previously unzipped content
get_ipython().system('unzip -o model_algo-2')


# Load the model parameters, and extract the weight matrix $W$ in the decoder.

# In[55]:


model = mx.ndarray.load('params')

W = model['arg:projection_weight']


# In[56]:


print(W)


# Visualize each topic as a word cloud. The size of each word is proportional to the pseudo-probability of the word appearing under each topic.

# In[57]:


get_ipython().system('pip install wordcloud')
import wordcloud as wc


# In[58]:


import matplotlib.pyplot as plt
word_to_id = dict()
for i, v in enumerate(vocab_list):
    word_to_id[v] = i

limit = 24
n_col = 4
counter = 0

plt.figure(figsize=(20,16))
for ind in range(num_topics):

    if counter >= limit:
        break

    title_str = 'Topic{}'.format(ind)

    #pvals = mx.nd.softmax(W[:, ind]).asnumpy()
    pvals = mx.nd.softmax(mx.nd.array(W[:, ind])).asnumpy()

    word_freq = dict()
    for k in word_to_id.keys():
        i = word_to_id[k]
        word_freq[k] =pvals[i]

    wordcloud = wc.WordCloud(background_color='white').fit_words(word_freq)

    plt.subplot(limit // n_col, n_col, counter+1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title_str)
    #plt.close()

    counter +=1


# # Congratulations!
# 
# You have completed this lab, and you can now end the lab by following the lab guide instructions.

# *©2023 Amazon Web Services, Inc. or its affiliates. All rights reserved. This work may not be reproduced or redistributed, in whole or in part, without prior written permission from Amazon Web Services, Inc. Commercial copying, lending, or selling is prohibited. All trademarks are the property of their owners.*
# 
