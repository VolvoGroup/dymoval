What is model validation?
=========================

You are developing a super product or process. Then, you test it. 
Then, you develop your work product a bit more. Then, you test it again... 
and you keep in iterating such a process over and over until the final deployment. 

This is the way things are typically developed. 

However, performing meaningful tests means to run your product (or service) in the *real-world* 
(aka *target environment*) has an 
associated cost, both in terms of money and in terms of time. 
Very often such cost also include personal stress.
  
A solution to alleviate such pain would be to run your tests in a *virtual environment* where there is a *model* 
of the *target environment* where your super-product or service shall be deployed.  

**Model validation** is the process of evaluating how good is your *model* and 
it is done by checking if your model "behaves" like the *target environment* it aims at representing.

A typical validation process consists in the following steps:

#. Design a set of experiments to be carried out on the target environment.
   The set of experiments consists in defining the set of stimuli (*input*) to the target environment,  

#. Run the experiments on the target environment and log its response. 
   The set of the used *input* signals and your system response is denoted as *measurement dataset* 
   (or, in short, just *dataset*),

#. Run the *same experiment* on your model, i.e. feed your model with the same stimuli as above and log its response. 
   We refer to the response of your model as **simulation results**,

#. Evaluate how "close" are the simulation results and the target environment output logged in the dataset 
   with respect to some validation metrics. 

ADD FIGURE.

If the results of step 4. are good, then we are safe to run tests on our model 
rather than running them on the target environment. 

.. note::

   Your model is valid only within the region covered by the dataset. 
   If you plan to use your model with data outside the dataset coverage region, then you have no guarantees that
   your model will behave like the target environment that you want to represent.



The cost saving when using models is clear, but there is no free lunch. 
In-fact, the challenge relies in the design of good models.

Nevertheless, although the design of good models is an art that cannot be completely automated, 
we can at least validate them automatically and here is where *dymoval* comes into play. 

Finally, let's conclude this Section by showing how the steps 1-4 can be applied through a simple example. 

   **Example**

   Assume that you have to develop some cool autonomous driving algorithm that shall be deployed in a car, 
   which represent your *target environment*.
   Now, assume that you already developed the model of a car where its input signals are *accelerator pedal position, 
   steering wheel position* and *road profile* whereas its output signals are *longitudinal speed* 
   and *lateral speed* of the vehicle. 
   You want to validate your model. 

   Steps 1-4 are carried out in the following way.

   #. Establish a driving route with sufficiently road slope variation. You decide to take a ride on that path by adopting a 
      nasty driving style that with sudden accelerations and abrupt steering movements.  
      
   #. You go out and drive according to plan and you log the *accelerator pedal position*, 
      the *steering wheel position* and the *road profile* time-series (input signals) along with the *longitudinal* and *lateral 
      speed* time-series of the vehicle (output signals). Such log-data represent your *dataset*. 
      **Note how input and output are separated in the dataset.**

   #. You feed your model with the dataset input signals and you observe your model output signals. 
      In other words, you feed your model with the *accelerator pedal position*, the *steering wheel position* and the *road profile* 
      time-series **logged during your ride** and you observe and store your model output 
      corresponding to the *longitudinal* and *lateral vehicle speed* into a *simulation results* data. 

   #. You compare the **logged** *longitudinal* and *lateral vehicle speed* time-series with the **simulated** 
      *longitudinal* and *lateral vehicle speed* time-series and you evaluate the results with respect to some validation metrics.

   If your model is "good" in accordance with the defined validation metrics, then you can safely use it for developing
   further your autonomous driving system, instead of testing on a real car - but it is great if you also test your stuff 
   in the target environment every once in a while. 

   But watch out because you have to take into account that your model is valid only within the region covered by the dataset. 
   This means that if the logged data cover only the accelerator pedal position in the range [0,40] %, steering angle 
   in the range [-2,2]° and the road profile was flat during all the collection data time, then you can trust your model 
   as long as you use it within those bounds!  

.. note::
   A small caution word on the nomenclature used: 
   we will interchangeably use the expressions *real-world system* and *target environment*. 
   This because what Control System engineers call *system* is often referred as *environment* by 
   Software Engineers.