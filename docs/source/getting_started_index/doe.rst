Design of Experiments (DoE)
===========================

When running experiments, it is important to stimulate the target environment in a way that we can extract
as much information as possible from it.

Good experiments shall stress the target environment as much as possible under different conditions.
This is important because a model is as trustworthy as the dataset used for validating it is *informative*.

    **Example**

    If we are developing the model of a car, we want to log sensors measurements 
    while driving within a wide range of speeds, with different accelerations profiles, 
    in different road and weather conditions and so on and so forth.

    If we log sensors measurements only when we are driving on a flat road and in the range 0-10 km/h 
    and by doing exactly the same manoeuvres over and over, then it would be hard to disagree 
    on that the collected dataset is poorly informative. 

 

At the current release, *dymoval* only stores the coverage region of the dataset and compute 
some statistics on it.
In this way, the user have an understanding under which conditions the developed model is trustworthy, 
provided that the validation results provides good figures.

.. note::
   In future releases we plan to further provide measures (Cramer-Rao Theorem? Fisher Matrix?) on the
   information level contained in a given dataset in within its coverage region.


A dataset covering a fairly large region 
won't necessarily imply information richness.
For example, this happens when you take a wide range of values but you stimulate your target environment 
only with constant inputs in within such a range. 
You would certainly have a dataset with a fairly large covered region but... it would contain little information.  

    **Example**

    With reference to the example above, you can imagine to drive the car only at constant speeds 
    in a range from 0 to 180 km/h on a flat road without never accelerating of braking.
    Your dataset will have a fairly large coverage region, but it will contain little information. 



How to design experiments that produce sufficiently informative datasets then?

Well, such an issue cannot be automatized but there is some theory behind it in the field of *Design of experiments (DoE)*,
feel free to google it for more details.
Due to that *DoE* cannot be automatized, it will not be included in dymoval, at least for now. 
