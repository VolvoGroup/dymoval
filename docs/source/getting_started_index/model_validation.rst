What is model validation?
=========================


*If you are already familiar with the concept of model validation you can skip this Section.*

Imagine that you are developing a super product or process. At some point, you need to test test it.
Then, based on the test outcome, you continue your development in a certain direction. 
Then, at some point you test your work-product again... 
and you keep in iterating such a process over and over until you get something good to deploy. 

This is the way things are typically developed. 

However, running tests in a *target environment*
(i.e. the *real-world environment* where your product - or service - shall be ultimately deployed) has an 
associated cost both in terms of money and in terms of time. 
Very often such a cost also include personal stress.

A solution to alleviate such pain would be to run your tests in a *virtual environment* where you have a *model* 
of your *target environment*. 
If the tests run in the virtual environment show good performance of your work-product, 
then you *should* get the same good performance when you move to the target environment.

Well, you *should* because the previous statement is true if, and only if, your virtual environment adequately 
represents the target environment and if it behaves similarly.

**Model validation** is the process of evaluating how good is the *model* of your *target environment*, 
i.e. it checks the similarity between your *virtual* and *target* environments through 
some validation metrics. 

A typical validation process consists in the following steps:

#. Design a set of experiments to be carried out on the target environment (aka *DoE*).
   The set of experiments consists in defining the set of stimuli (*input*) to be given to the target environment,  

#. Run the experiments of point 1. on the target environment and log its response. 
   The set of the *input* signals along with the response of your target environment is denoted as *measurement dataset* 
   (or, in short, just *dataset*),

#. Run exactly the *same experiment* defined in point 1. on your model and you log its response. 
   We refer to the response of your model as **simulation results**,

#. Evaluate how "close" are the simulation results of point 3. and the logged response of point 2. 
   with respect to some validation metrics. 

.. figure:: ../figures/ModelValidation.svg
   :scale: 50 %

   The model validation process.  In this picture the validation method only returns a pass/fail value but in general it returns the evaluation of some model quality metrics.  

If the results of step 4. are good, then you can safely keep in developing and test in the virtual environment. 
Most likely things will work in the target environment as well - but it is good practice to verify that every once in a while.

Keep in mind that *"all models are wrong, but some are useful."* ;-).


However, it is worth noting that your model is valid only within the **region covered** by the dataset. 
If you plan to use your model with data outside the dataset coverage region, then you have no guarantees that
things will work in target environment as they worked in the virtual environment.


   **Example**
   Let's show how steps 1-4 can be applied through a simple real-world example. 

   Assume that you are developing some cool autonomous driving algorithm that shall be deployed in a car, 
   which represent your *target environment*.

   Assume that you developed the model of a car where its **input** signals are:

   #. *accelerator pedal position*, 
   #. *steering wheel position* and 
   #. *road profile*, 
   
   whereas its **output** signals are:

   #. *longitudinal speed* and 
   #. *lateral speed*.
   
   Next, you want to validate your model. 

   Steps 1-4 are carried out in the following way:

   #. You choose a driving route with sufficiently road slope variation. You decide to take a ride on that path by adopting a 
      nasty driving style with sudden accelerations and abrupt steering movements. Congrats! You just made a Design of Experiment (DoE).  
      
   #. You take a ride with the target vehicle and you drive according to the DoE performed in the previous step. 
      You log the input signals (i.e. the *accelerator pedal position*, 
      the *steering wheel position* and the *road profile* time-series) along with the output signals (i.e. *longitudinal* and *lateral 
      speed* time-series) of the vehicle while driving. Such logs represent your *dataset*. 
      Note how input and output are separated.

   #. Feed your model with the input signals that you logged during the drive and log your model output 
      corresponding to the *longitudinal* and *lateral vehicle speed* dynamics. 
      
   #. Compare the *longitudinal* and *lateral vehicle speed* time-series logged during the actual drive with the *simulated results* with respect to some validation metrics.


   You haven' finished yet. 
   In-fact, when you develop and validate a model, you should also consider the coverage region of the model along with the validation results. 

   If you logged data only in the *accelerator pedal position* range [0,40] %, the *steering angle* 
   in the range [-2,2]° and the *road profile was flat* for all the time, then you have to ship such an information
   along with your model to a potential model user.

The cost saving when using models is clear, but there is no free lunch. 
In-fact, the challenge relies in the design of good models.

Nevertheless, although the design of good models is an art that cannot be completely automated, 
we can at least validate them automatically and here is where *Dymoval* comes into play. 
In-fact, *Dymoval* will not help you in developing any model at all, it will just tell you 
if your models are good or not *after* you developed them. 

.. note::
   A small caution on the nomenclature used: 
   we will interchangeably use the expressions *real-world system* and *target environment*. 
   This because what Control System engineers call *system* is often referred as *environment* by 
   Software Engineers.
