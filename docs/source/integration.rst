CI/CD integration
=================

*Dymoval* can be used for unit-testing your models and it can be further used in development pipelines like those 
provided for example by Jenkins or GitLab.

Unit-test
---------

The development of large models is typically done by breaking it down in smaller components.

For example, if you are developing the model of a car, you may need to develop the model of the engine, 
the model of the tires, the model of the gearbox - this really depends on your need and can be highly debated - 
and then you integrate them.

However, smaller components are models themselves and therefore they can be validated against some dataset through *dymoval*.
This means that you can use *dymoval* for unit-testing single components.

CI/CD integration
-----------------

A traditional software development workflow consists in pushing your software changes towards a repo  
where there is some source automation server (like Jenkins or GitLab) that automatically assess if your changes 
can be integrated in the codebase or not.

Very often, the process of developing models goes along the same line: you typically have a Version Control System (VCS) 
that track your model changes...

... but there are no automated mechanism that test your model.
Checking that *"everything still works"* is typically done manually and if your changes cane be 
integrated or not is at reviewer discretion. 
Not robust, nor democratic.  

The ideal scenario would be to automatically test your model changes every time they 
are pushed towards the remote repo, as it happens in traditional software development.
If on the one hand you are developing *models* - and not, loosely speaking, code -  
on the other hand, testing a model just means to *validate* it.

Here is where *dymoval* comes into play: you can exploit its API to write scripts that can be automatically executed by 
automation tools and you can automatically get an answer if your changes can be integrated or not 
depending if the validation metrics evaluation meet some criteria (for example if they pass a chi-squared test).




