We analyze the complexity overhead and a learning curve associated with the transition from quick-and-dirty machine learning experiments to large-scale production-grade models with the recently released [Amazon SageMaker](https://aws.amazon.com/sagemaker/) and open-source project [Studio.ML](studio.ml)

Virtually every domain of human expertise is facing a rapid increase in the integration of machine learning solutions and tools. With the number of machine learning experiments and models growing, data science teams in big and small companies realize the need for a unifying framework, one that lets data scientists build on top of their own models and experiments as well as leverage the efforts of other community teams and members in an efficient manner. The "data science github" concept, however, faces multiple challenges (mainly related to the usage of large datasets, large amounts of computing resources and custom hardware). Multiple attempts to solve these issues have been made, the most well-known are [Google Cloud ML](https://cloud.google.com/ml-engine/) and the recently released Amazon SageMaker. 

In this blog series, we’ll build several models ranging from very simple toy examples to state-of-the-art deep neural networks presented at NIPS 2017. We’ll make the model training reproducible in the cloud using modern frameworks, and show why Sentient Technologies continues to support the open-source project Studio.ML. 

# Part 1. K-means with Amazon SageMaker and Studio.ML.

The story revolves around a fairly simple exercise (chosen from the SageMaker getting started guide to ensure my lack of knowledge with the SageMaker is not affecting the results) - building a K-means model of MNIST data. 

For  people not familiar with K-means - it is an unsupervised learning algorithm (i.e. we'll feed the algorithm the images of digits from MNIST, but not the labels) that searches for centers of clusters. Each step consists of assigning data samples (in our case, 28x28 grayscale MNIST images -> 784 dimensional vector samples) to clusters, and then moving the cluster centers to the averaged coordinates of data samples in each cluster. At the prediction time, we find the cluster center closest to the input image. 

## K-Means with SageMaker

To get started with SageMaker, one needs to create a notebook instance. SageMaker comes with a very user-friendly UI that walks you through the process.

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_0.png)

Notebook comes with an AMI (instance image) that has all common ML libraries pre-installed. 

Once the notebook creation request is submitted, AWS provisions and sets up the instance; this process  usually takes 4-6 minutes. When the notebook is created, one clicks "Open" in the SageMaker console to open a familiar jupyter notebook window. 

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_1.png)

The first cell deals with imports and downloading the MNIST data, while the second cell converts and uploads the data to the s3 bucket (the SageMaker training routine assumes data resides within S3). 

Now we are ready to launch training: 

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_2.png)

Okay, so now we have a trained model. How can we use / validate the model? The model artifacts do not seem to be easy to introspect; however, we can serve and predict using the served model: 

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_3.png)

Note that deploying the model takes a little while (in this case, 12 minutes - longer than the training itself) which is ok if the deployed model will be used in production, but sounds excessive for simple model validation. 

After predicting, we can visualize the images from various clusters. Learning is unsupervised, so the digits in the clusters do not have to correspond to the cluster number. We see that images that look like 6 are assigned to cluster number 0. 

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_4.png)

Another limitation of the prediction via a call to an endpoint is the maximum number of images one can process at a time:

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_5.png)

## K-Means with Jupyter Notebook

Let's look into how would we run the same process without SageMaker. We'll need python packages urllib (to download the data), sklearn (to actually perform the k-means), matplotlib (for visualization) and jupyter (for the notebook). If you have done any machine learning in python before, chances are you have all of those already. Otherwise, they can be installed via entering

<table>
  <tr>
    <td>pip install urllib sklearn matplotlib jupyter</td>
  </tr>
</table>


in the command line.  Then the training is achieved with the following cells

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_6.png)

The prediction is done as follows (this time images that look like 6 got assigned to cluster 1)

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_7.png)

Note that when using sklearn, model introspection is very simple. For example, the following code will show the images that correspond to the cluster centroids. The cluster centroids are more blurry than the actual digit images because they are the means of all images in the cluster. 

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_8.png)

So far, we can see that for simple tasks, using jupyter notebook with the right package (in this case, sklearn) can be much simpler and more flexible than using the SageMaker. What about complicated tasks? If the complexity is in the amount of data, there is no doubt that SageMaker will shine - after all, the careful reader noticed that we were using c4.8xlarge instance for training, which can crunch numbers much faster than my laptop. However, what if the complexity is also in a new algorithm (either training algorithm, or model wrapper, or both)? In SageMaker you need to build your own container (either the same for training and inference, or two different ones) following specific guidelines, then register and use the container ([https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb)). The existing primitives help you along the way; but it is still painful especially when making containers to be gpu-compatible ([http://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html](http://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html)). The biggest pain point for me though would be debugging the training algorithm and navigating the way it is wired up within the container. By contrast, in the regular jupyter notebook, as soon as something goes wrong, you can drop into pdb, inspect variables and see immediately what is going on. We have already seen a machine learning analog of debugging - model introspection - being easily done in jupyter notebook, but being fairly hard (if not impossible) in SageMaker. 

But wait you might say, how is it fair to compare plain jupyter with SageMaker? Anyone can use a screwdriver, but operating a CNC milling machine requires training. Besides, surely the fact that one can carry a screwdriver in a pocket does not make the screwdriver more useful than the milling machine? What about serving the models? What about training on custom hardware? What about hyperparameter optimization? What about the model provenance? 

And finally, why is Studio.ML in the title of this blog? 

The answer to all of this questions is the following. Studio.ML gives you all of the above (serving, custom hardware in the cloud, hyperparameter optimization, model provenance) without leaving the comfort zone of the locally (or where ever your preference is)-running jupyter notebook or python command line. In a sense, it is a CNC milling bit for your pocket data science screwdriver. 

## K-Means with Studio.ML

Let us install studioml package via

<table>
  <tr>
    <td>pip install studioml</td>
  </tr>
</table>


Then, we use the same jupyter notebook as in the last exercise (K-means with sklearn), and add a single line to the imports section that imports cell magics from studio:

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_9.png)

To start an experiment with studio, we simply add a cell magic %%studio_run to the notebook cell:

Technically, studio.ml returns all the variables created in the cell, so I also erase train_set and test_set variables to prevent them being returned)

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_10.png)

The link in the beginning of the experiment sends us to a central repo of experiments, to the experiment page that shows experiment progress, artifacts, and list of python packages necessary to reproduce the experiment.

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_11.png)

The code in the cell runs in 6 minutes (slightly longer than plain sklearn due to, mainly, compressing, storing in the cloud, and returning the validation data; still 3 minutes faster than using a dedicated training instance in SageMaker)

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_12.png)

If the training is more heavy-weight and actually requires extended usage of powerful instances, we can request them from the cloud by modifying studio_run magic. For example, 

<table>
  <tr>
    <td>%%studio_run --cloud=ec2spot --cpus=16</td>
  </tr>
</table>


will run the training on a spot instance with 16 cpus (the c4.4xlarge instance will be selected for that).  The cool part is that once the training is complete, the rest of the notebook code is exactly the same as it used to be, including prediction and displaying cluster centers.

This time 6-like digits accidentally ended up in a cluster with a correct number

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_13.png)

So far, we have shown that Studio.ML provides zero-overhead experiment provenance and provisioning of custom hardware, including spot instances, different clouds (for now Studio.ML has support for Amazon EC2 and Google Cloud). What about model serving? Studio.ML does come with model serving primitives for Keras, but not for sklearn (not yet, at least); however, the process of making one is very transparent. 

Let us write the following python script to be used as a model wrapper and call in kmeans_serve.py

<table>
  <tr>
    <td>import pickleimport osdef create_model(modeldir):    with open(os.path.join(modeldir, 'kmeans.pkl.gz')) as f:        kmeans = pickle.loads(f.read())    def model_function(input_dict):        predictions = kmeans.predict(pickle.loads(input_dict['input_data']))        return {"output_data": pickle.dumps(predictions)}    return model_function</td>
  </tr>
</table>


which loads the pickled model kmeans from file kmeans.pkl in the directory modeldir (passed as an argument) and returns a function that, given a dictionary {"input_data": <pickled_input_data>} performs inference and returns the results as a dictionary {"output_data": <pickled_output_data>}. This approach can be used with any model that consumes pickleable data. 

We then run the following command:

<table>
  <tr>
    <td>studio serve 1513115524_jupyter_7afb38f0-9918-48b8-9921-0d29f44f421d --wrapper=kmeans_serve.py</td>
  </tr>
</table>


where 1513115524_jupyter_7afb38f0-9918-48b8-9921-0d29f44f421d is the key of the experiment returned in the output of %%studio_run cell magic. 

This command will serve the model locally, so in our notebook the following command generates the predictions:

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_14.png)

As before, the predictions can be visualized as follows: 

![image alt text]({{ site.baseurl }}/images/zerooverhead_part1/image_15.png)

Serving does support EC2 and GCloud, so, for example, a command 

<table>
  <tr>
    <td>studio serve 1513115524_jupyter_7afb38f0-9918-48b8-9921-0d29f44f421d --wrapper=kmeans_serve.py --cloud=ec2</td>
  </tr>
</table>


will serve the model on an ec2 instance. 

Served models have an automatic expiration time (by default, one hour) so one does not have to worry about runaway instances. 

## Summary

We compared the experience of building a simple k-means model on MNIST dataset in SageMaker, jupyter notebook, and studio.ml. SageMaker provides a rich toolset out of the box, however, those tools are somewhat out of line with the natural jupyter notebook way; extension of the SageMaker toolbox to meet custom needs seems fairly complicated. In contrast, Studio.ML can seamlessly extend jupyter notebooks to provide the experiment provenance, training and / or inference on custom cloud compute (including spot instances) and serving. 

In our next blog, we’ll look into training the Fader Network (([https://arxiv.org/pdf/1706.00409.pdf](https://arxiv.org/pdf/1706.00409.pdf)) - a neural network that can interpolate and swap between the attributes of an image such as presence/absence of glasses, open/closed eyes etc. The Fader Networks require a lot of computational resources to train, and as such, they are a tempting and yet difficult to chew fruit for data scientists outside large corporations such as Google or Facebook, which makes them a good demo for the full power of machine learning provenance frameworks such as the SageMaker or Studio.ML. 

