Design of Experiments (DoE)
===========================

When running experiments, it is important to stimulate the target environment in a way that we can extract
as much information as possible from it.

Good experiments shall stress the target environment as much as possible under different conditions.
This is important because a good model assessment heavily depends on how *informative* is the dataset 
that we use in the validation phase.

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

In future releases we plan to further provide measures (Cramer-Rao Theorem? Fisher Matrix?) about the
information level contained in a given dataset in within its coverage region.

This is justified by that having a dataset covering a fairly large region 
won't necessarily imply that it is also information rich.
For example, this can happen when you stimulate your target environment only with constant inputs 
in a wide range of values. 
You would get a dataset with a fairly large covered region but that would contain very little information.  

    **Example**

    With reference to the example above, you can imagine to drive the car only at constant speeds 
    in a range from 0 to 180 km/h on a flat road without never accelerating of braking.
    Your dataset will have a fairly large coverage region, but it will contain little information. 



What is left out in this part, and will not be included in *dymoval*, is how to design experiments 
that produce sufficiently informative datasets.
This issue cannot be automatized and it is addressed in the field of *Design of experiments (DoE)*,
feel free to google it for more details.
