# **1. AI for Medical Diagnosis**

## **1.1 Welcome to the AI for Medicine Specialization**

### **1.1.1 Welcome to the Specialization**

Given an image of a chest X-ray, so unstructured image data, can we train a neural network to diagnose whether or not a patient has pneumonia? Or given structure data such as the patient's lab results, can we train a decision tree to estimate the risk of heart attack? By working on these concrete problems, we also see a lot of the practical aspects of machine learning from how they deal with imbalanced data sets, to how to work with missing data, to picking the right evaluation metric. In machine learning, we often default to classification accuracy as the metric. But for many applications, that's not the right metric.

How do we choose a more appropriate one? AI for medicine is taking off all around the world right now. So this is actually a great time for you to jump in and try to have a huge impact. Maybe you can be the one to invent something that saves a lot of patients' lives.

This is a 3-course specialization. In the 1. course, we learned about building machine learning models for diagnosis. Diagnosis is about identifying disease. In the 1. course, we will build an algorithm that will look at a chest X-ray and determine whether it contains disease. We will also build another algorithm that will look at brain MRIs and identify the location of tumors in those brain MRIs. The 2. course will be on predicting the future health of the patients, which is called prognosis. In the 2. course, we will learn how to work with structured data. Let's say, we have a patient's lab values and their demographics, and use those to predict the risk of an event, such as their risk of death or their risk of a heart attack. In the 3. course, we learned about AI for treatment. That is, for the process of medical care and also for information extraction, getting information out of medical texts.

In Course 2, we will learn how to use machine learning models to be able to estimate what the effect of a particular treatment would be on a patient. We will also learn about the application of AI to text for particular tasks like question answering and extracting labels from radiology reports.

Diagnosis means the process of determining which disease or condition explains the person's symptoms, signs, and medical results. In particular, we will be learning how to build and evaluate deep-learning models for the detection of disease from medical images. Just in the 1. week, we build a deep learning model that can interpret chest X-rays to classify different disease causes. In the 2. week, we will implement evaluation methodologies to assess the quality of your model. In the 3. week, we use image segmentation to identify the location and boundaries of brain tumors in MRI scans.

### **1.1.2 Demo**

We will build a deep learning model that can interpret chest X-rays to classify different diseases. A demo of what your model will be able to do. We have a chest X-ray. Chest X-ray is the most commonly performed imaging exam in the world and used to diagnose and treat a variety of diseases. This is a chest X-ray of a patient who has fluid in their lungs. When we train our model it will be able to pick up this abnormality. Let's see it in action.

So here I've just run this image of the chest X-ray to a deep learning model on my phone. The model processes this chest X-ray and recognizes that this patient's X-ray is abnormal. More specifically, when I scroll down, it's found that the patient likely has fluid in their lungs. This condition is called edema. In course 3, we will learn about how we can generate these heat maps that show where in the image the model is finding evidence of the disease. 

## **1.2 Applications of Computer Vision to Medical Diagnosis**

### **1.2.1 Medical Image Diagnosis**

We'll be diving straight into building a deep learning model for the task of chest x-ray classification. Many of the ideas that we will learn through this example are broadly applicable across many medical imaging tests. We'll start by looking at three examples of medical diagnostic tasks where deep learning has achieved incredible performance. We'll then jump into the training procedure for building AI models for medical imaging. Finally, we'll look at the testing procedure for evaluating the performance of these models on real data.

Our first example is in dermatology. Dermatology is the branch of medicine dealing with the skin. One of the tasks dermatologists perform is to look at a suspicious region of the skin to determine whether a mole is skin cancer or not. Early detection could likely have an enormous impact on skin cancer outcomes. The five-year survival rate of one kind of skin cancer drops significantly if detected in its later stages. In this study, an algorithm is trained to determine whether a region of skin tissue is cancerous or not.

Using hundreds of thousands of images and labels as input, a CNN can be trained to do this task. We will look at the training of such an algorithm.

Once the algorithm has been trained, the algorithms predictions can be evaluated against the predictions of human dermatologists on a new set of images. In this study, it was found that the algorithm performed as well as the dermatologists did. Don't worry too much about the interpretation of this graph right now. In a future week of the course, we will look at the evaluation using such a curve. The main conclusion we will be able to draw from this graph is that the algorithms prediction accuracy was comparable to the prediction of human dermatologists.

### **1.2.2 Eye Disease and Cancer Diagnosis**

Our second example is in ophthalmology, which deals with the diagnosis and treatment of eye disorders. One well-known study in 2016 looked at retinal fundus images, which photographed the back of the eye. One disease or pathology to look at here is diabetic retinopathy (DR), which is damage to the retina caused by diabetes and is a major cause of blindness. Currently, detecting DR is a time-consuming and manual process that requires a trained clinician to examine these photos. In this study, an algorithm was developed to determine whether patients had diabetic retinopathy by looking at such photos.

This study used over 128,000 images of which only 30% had diabetic retinopathy. We'll look at this data imbalanced problem, which is prominent in medicine and in many other fields with real-world data. We will see some methods for tackling this challenge. Similarly to the previous study, this study showed that the performance of the resulting algorithm was comparable to ophthalmologists. In the study, a majority vote of multiple ophthalmologists was used to set the reference standard or ground truth, which is a group of experts, best guess of a right answer. Later this week in the course, we'll look at how ground truth can be set in such medical AI studies.

Our third example is in histopathology, a medical specialty involving examination of tissues under the microscope. One of the tasks that pathologists do is look at scanned microscopic images of tissue called whole slide images and determine the extent to which a cancer has spread. This is important to help plan treatment, predict the course of the disease, and the chance of recovery. In one study in 2017, using only 270 whole slide images, AI algorithms were developed and then evaluated against pathologists. It was found that the best algorithms performed as well as the pathologists did.

In histopathology, the images are very large and cannot be fed directly into an algorithm without breaking them down. The general setup of these studies is that instead of feeding in one large, high resolution digital image of the slide, several patches are extracted at a high magnification and used to train a model. These patches are labeled with the original label of the whole slide image and then fed into a deep learning algorithm. In this way, the algorithm can be trained on hundreds of thousands of patches. In this course, you will apply a similar idea of breaking down a large image into smaller images for model training to the task of brain tumor segmentation.

## **1.3 Handling Class Imbalance and Small Training Sets**

### **1.3.1 Building and Training a Model for Medical Diagnosis**

Now that we've seen some of the cutting-edge applications of deep learning to medical image classification problems, we'll look at how we can build our own deep learning model for the medical imaging task of using chest X-rays to detect multiple diseases with a single model. We'll walk through the process of training a model for chest X-ray interpretation and look at the key challenges that we will face in this process, and how we can go about successfully tackle.

We'll start by looking at the task of chest X-ray interpretation. The chest X-ray is one of the most common diagnostic imaging procedures in medicine with about 2 billion chest X-rays that are taken for a year. Chest X-ray interpretation is critical for the detection of many diseases, including pneumonia and lung cancer which affect millions of people worldwide each year. A radiologist who is trained in the interpretation of chest X-rays looks at the chest X-ray, looking at the lungs, the heart, and other regions to look for clues that might suggest if a patient has pneumonia or lung cancer or another condition.

Let's look at one normal abnormality, called a mass, looks like. And I'm not going to first define what a mass is but let's look at three chest X-rays that contain a mass and three chest X-rays that are normal. I can show you a new chest X-ray and ask you to identify whether there is a mass. We might be able to correctly identify that this chest X-ray contains a mass. Here's the mass that might look similar to things that we see in these images, but not similar to anything that you see in these images. The way we are learning is very similar to how we're going to teach an algorithm to detect mass. For our own reference, a mass is defined as a lesion, or in other words damage of tissue, seen on a chest X-ray as greater than 3 centimeters in diameter. Let's see how we can train our algorithm to identify masses.

### **1.3.2 Training, Prediction, and Loss**

During training, an algorithm is shown images of chest X-rays labeled with whether they contain a mass or not. The algorithm learns using these images and labels.

The algorithm eventually learns to go from a chest X-ray input to produce the output of whether the X-ray contains mass. This algorithm can go by different names: “Deep Learning Algorithm” or “Model” or “Neural Network” or “Convolutional Neural Network”.

The algorithm produces an output in the form of scores, which are probabilities that the image contains a mass. The probability that this image contains a mass is outputted to be 0.48, and the probability for this image is outputted to be 0.51. When training has not started, these scores, these probability outputs are not going to match the desired label. Let's say the desired label for mass is 1, and for normal is 0. 0.48 is far off from 1 and 0.51 is far off from the desired label of 0. We can measure this error by computing a loss function. A loss function measures the error between our output probability and the desired label. We'll look at how this loss is computed soon enough. Then, a new set of images and desired labels is presented to the algorithm as it learns to produce scores that are closer to the desired labels over time. Notice how this output probability is getting closer to 1, and this output probability is getting closer to 0.


Image Classification and Class Imbalance
Typically, in medical AI examples, hundreds of thousands of images are shown to the algorithm.
This is a typical setup for image classification, which is a core task in the computer vision field where a natural image is input to an image classification algorithm, which says what is the object contained in the image. You may have seen deep learning algorithms that can do this. Our example of chest X-ray classification is similar in many ways to the image classification setup. There are a few additional challenges which make training medical image classification algorithms more challenging, which we'll cover next.
We'll talk about three key challenges for training algorithms on medical images:
-	the class imbalance challenge,
-	the multitask challenge,
-	the dataset size challenge.
For each challenge, we'll cover one or two techniques to tackle them.
Let's start with the class imbalance challenge. There's not an equal number of examples of non-disease and disease in medical datasets. This is a reflection of the prevalence or the frequency of disease in the real-world, where we see that there are a lot more examples of normals than of mass, especially if we're looking at X-rays of a healthy population. In a medical dataset, you might see 100 times as many normal examples as mass examples.
Binary Cross Entropy Loss Function
This creates a problem for the learning algorithm would seize mostly normal examples. This yields a model that starts to predict a very low probability of disease for everybody and won't be able to identify when an example has a disease.
Note: log with base 10 (lg) is being used to calculate the loss. One other popular choice is to use ln.
How can we trace this problem to the loss function that we use to train the algorithm? How can we modify this loss function in the presence of imbalanced data. This loss over here is called the binary cross-entropy loss and this measures the performance of a classification model whose output is between 0 and 1.
Let's look at an example to see how this loss function evaluates. We have an example of a chest x-ray that contains a mass, so it gets labeled with one and the algorithm outputs a probability of 0.2. The 0.2 is the probability according to the algorithm of P(Y=1|X), the probability that this example is a mass. We can apply the loss function to compute the loss on this example. Our label is 1, so we're going to use the first term {-logP(Y=1|X) if y=1}. Our loss is -log and then we're going to take the algorithm output, 0.2. This evaluates to 0.70.
L = -log(0.2) = 0.70  This is the loss that the algorithm gets on this particular example.
Let's look at another example. This time a non-mask example, which would have a label of 0. Our algorithm outputs a probability of 0.7. We're going to use {-logP(Y=0|X) if y=0}of the loss rate because y=0. The loss is going to be -log of the term P(Y=0|X). We can get P(Y=0|X) using P(Y=1|X). The way we can compute this quantity from that one is by recognizing that P(Y=0|X) that an example is 0 is 1-P(Y=1). An example as either mass or not. The algorithm says 70% probability that something is mass, then there's 30% probability it's not. We're going to plug in 1 - 0.7 = 0.3 and this expression evaluates to 0.52.
L= -log(1-0.7) = -log(0.3) = 0.52
Impact of Class Imbalance on Loss Calculation
We've seen how the loss is applied to a single example. Let's see how it applies to a bunch of examples.
We have 6 examples that are normal, and 2 examples that are mass.
P1, P2, P3, P5, P6, P8 are “Normal”.
P4 and P7 are “Mass”.
P2, P3, P4 are the patient IDs. When the training hasn't started, let's say the algorithm produces an output probability of 0.5 for all of the examples, the loss can then be computed for each of the examples.
For a normal example, we're going to use -log(1-0.5)=0.3.
For a mass example, we're going to use -log(0.5)=0.3.
The total contribution to the loss from the mass examples comes out to 0.3x2=0.6.
The total loss from the normal example, comes out to 0.3x6=1.8.
Notice how most of the contribution to the loss comes from the normal examples, rather than from the mass examples. The algorithm is optimizing its updates to get the normal examples, and not giving much relative weight to the mass examples. In practice, this doesn't produce a very good classifier. This is the class imbalance problem. The solution to the class imbalance problem is to modify the loss function, to weight the normal and the mass classes differently.
wp will be the weight we assign to the positive or to the mass examples, and wn to the negative or normal examples.
Let's see what happens when we weight the positive examples more. We want to weight the mass examples more, such that they can have an equal contribution overall to the loss, as the normal examples.
Let's pick 6/8 as the weight we have on the mass examples, and 2/8 as the weight we have on the normal examples. If you sum up the total loss from the mass example, we get 0.45, and this is equal to the total loss from the normal examples. In the general case, the weight we'll put on the positive class will be the number of negative examples over the total number of examples. In our case, this is 6 normal examples over 8 total examples. The weight we'll put on the negative class will be the number of positive examples over the total number of examples, which is 2/8. With this setting of wp and wn, we can have over all of the examples for the loss contributions from the positive and the negative class to be the same. This is the idea of modifying the loss using weights, in this method that's called the weighted loss, to tackle the class imbalance problem.

