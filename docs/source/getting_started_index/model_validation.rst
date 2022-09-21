What is model validation?
=========================

Imagine that you are developing a super product or process. At some point, you need to test test it.
Then, based on the test outcome, you continue your development in a certain direction. 
Then, at some point you test again... 
and you keep in iterating such a process over and over until you get something good to deploy. 

This is the way things are typically developed. 

However, to run tests in a *target environment*
(i.e. the *real-world environment* where your product - or service - shall be ultimately deployed) has an 
associated cost both in terms of money and in terms of time. 
Very often such cost also include personal stress.

A solution to alleviate such pain would be to run your tests in a *virtual environment* where you have a *model* 
of your *target environment*. 
If the tests of your work-product in the virtual environment show good performance, 
then you *should* get the same performance when you perform the same tests in the target environment.
Well, you *should* because the previous statement is true if, and only if, your virtual environment adequately 
represents the target environment and if it behaves similarly.

**Model validation** is the process of evaluating how good is the *model* of your *target environment*, 
i.e. it checks the similarity between your *virtual* and *target* environments through 
some validation metrics. 

More precisely, a typical validation process consists in the following steps:

#. Design a set of experiments to be carried out on the target environment.
   The set of experiments consists in defining the set of stimuli (*input*) to be given to the target environment,  

#. Run the experiments of point 1. on the target environment and log its response. 
   The set of the *input* signals along with the response of your target environment is denoted as *measurement dataset* 
   (or, in short, just *dataset*),

#. Run exactly the *same experiment* defined in point 1. on your model and you log its response. 
   We refer to the response of your model as **simulation results**,

#. Evaluate how "close" are the simulation results of point 3. and the logged response of point 2. 
   with respect to some validation metrics. 

ADD FIGURE.

If the results of step 4. are good, then you can safely keep in developing and test in the virtual environment. 
Most likely things will work in the target environment as well - but it is good practice to verify that every once in a while.
Keep in mind that *"all models are wrong, but some are useful."* ;-).


However, it is worth noting that your model is valid only within the region covered by the dataset. 
If you plan to use your model with data outside the dataset coverage region, then you have no guarantees that
things will work in target environment as they work in the virtual environment.



Let's make an example showing how the steps 1-4 can be applied through a simple real-world example. 

   **Example**

   Assume that you are developing some cool autonomous driving algorithm that shall be deployed in a car, 
   which represent your *target environment*.

   Assume that you already developed the model of a car where its **input** signals are

   #. *accelerator pedal position*, 
   #. *steering wheel position* and 
   #. *road profile*, 
   
   whereas its *output* signals are 

   #. *longitudinal speed* and 
   #. *lateral speed*,
   
   of the vehicle. 
   Next, you want to validate your model. 

   Steps 1-4 are carried out in the following way

   #. Establish a driving route with sufficiently road slope variation. You decide to take a ride on that path by adopting a 
      nasty driving style that with sudden accelerations and abrupt steering movements.  
      
   #. Take a ride with the target vehicle and drive according to plan while logging the input signals (i.e. the *accelerator pedal position*, 
      the *steering wheel position* and the *road profile* time-series) along with the output signals (i.e. *longitudinal* and *lateral 
      speed* time-series) of the vehicle. Such log-data represent your *dataset*. 
      Note how input and output are separated.

   #. Feed your model with the input signals *logged during your ride* and log your model output 
      corresponding to the *longitudinal* and *lateral vehicle speed* into a *simulation results* data. 
      outputs (*longitudinal* and *lateral vehicle speed* time-series) and you evaluate the results with respect to some validation metrics.

   You haven' finished yet. 
   In-fact, when you develop and validate a model, you should ship the coverage region of our model along with the validation results. 

   If you logged data only in the *accelerator pedal position* range [0,40] %, the *steering angle* 
   in the range [-2,2]° and the *road profile was flat* for all the time, then you have to ship such an information
   along with your model so that the user knows in which region he/she can trust your model.

The cost saving when using models is clear, but there is no free lunch. 
In-fact, the challenge relies in the design of good models.

Nevertheless, although the design of good models is an art that cannot be completely automated, 
we can at least validate them automatically and here is where *Dymoval* comes into play. 
In-fact, *Dymoval* wil not help you in developing any model at all, it will just tell you 
if your modes are good or not. 

.. note::
   A small caution on the nomenclature used: 
   we will interchangeably use the expressions *real-world system* and *target environment*. 
   This because what Control System engineers call *system* is often referred as *environment* by 
   Software Engineers.