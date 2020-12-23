# How-to-Talk-2D-Generative-Modeling

December 23, 2020

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!

Hire me! 😊

- We now continue our development
of the generative approach to classification.
So last time, we looked
at the one-dimensional Gaussian distribution,
and we saw how to build a classifier
from it using the generative approach.
Today, we'll look at the Gaussian in two dimensions.
One of the exciting things about this distribution is
that it allows us to model the dependence between features.
So, for example, if we are trying to predict
whether somebody's ill based on their cholesterol level
and their blood pressure, we don't have
to pretend that these two features,
cholesterol level and blood pressure, are independent.
We can actually model the dependence between them
in order to get a more accurate prediction.
So our running example is the winery prediction problem.
And if you recall, the situation here is
that we have a bottle of wine.
We have bottles of wine, and from each bottle,
we measure 13 visual and chemical features.
And we want to use these to predict
which winery this bottle came from,
winery one, two or three.
Last time, we used just one out of the 13 features,
and we found that using that one feature alone,
we could get an error rate of about 29%.
Today, we'll use two features, two out of the 13,
and we'll see that the error rate drops quite dramatically.
So just to remind you about this data set,
we had a training set obtained
from 130 bottles, so 130 data points.
43 of them were from winery one,
51 from winery two and 36 from winery three.
And from each bottle, we extracted 13 features
that makes sense to wine experts.
There was also a separate test set with 48 labeled points.
Now last time, the feature that we used was alcohol.
This time we'll use a second feature, flavonoids.
These are a family of chemical compounds
that show up in wine.
Okay, so these are the two features we're gonna use.
And let's just sort of think at a high level first
about why it might be helpful to throw in a second feature.
So what happened last time is we just had
this one feature, alcohol.
And we fit a Gaussian to the alcohol levels from winery one,
another Gaussian to the alcohol levels from winery two
and a third one to the alcohol levels from winery three.
Looking at the three Gaussians, it turned out
that they aren't really very well separated from each other.
In fact, they're kind of on top of each other.
And this is why this one feature alone turned out
to not be a great basis for doing classification.
So let's see what happens when we throw in a second feature.
So what I've shown here is a scatterplot.
Each point is one of the 130 training points.
The red ones are from winery one.
The black ones are from winery two.
The green ones are from winery three.
And for each of the training points, I've shown two
of the features, alcohol level and flavonoids, okay.
So each point has been plotted in two dimensions.
And you can see that there is actually quite a bit
of separation between them.
So this makes us hopeful.
It gives us hope that these two features might be the basis
for better classification.
So what we're gonna do is to follow the generative approach
and to fit a distribution to the data from winery one,
a separate distribution to winery two
and a separate distribution to winery three.
And the distributions are gonna turn out
to be something like this, okay.
This might seem mysterious.
What are these ellipses?
So I will explain in a second.
But roughly, this is the picture.
And it turns out when you do this,
the error rate drops quite substantially,
from 29% to just 8%, okay.
So all this is great, but what are these distributions,
and what are these ellipses?
So let's hone in on one of them.
Let's just look at the red one,
which is from winery number one.
Let's just look at those specific points.
So when we fit a distribution to those,
when we fit a Gaussian distribution
to those 43 training points,
this is the distribution we get, okay.
So what is this thing?
Well, so it's a distribution over the 2D plane, okay.
So it's a distribution that assigns a value
to every possible alcohol level
and every possible flavonoid level.
So at the base over there,
you see the alcohol and flavonoid values.
That's a 2D plane, and the height is the density
at that point, okay.
This is what a bivariate Gaussian distribution looks like.
Now for me, personally, it's a little hard
to interpret these 3D pictures, so I much prefer
to draw it like this, okay.
So what I've drawn here
in the red are the original training points,
the 43 training points,
the 43 two-dimensional points from winery one.
Using these points, we fit a Gaussian distribution,
and this distribution has two sets of parameters.
It has a mean, and it has a covariance matrix.
The mean is just a 2D point.
It is the center of the distribution.
It is the point of highest density.
So we're fitting a distribution, and there's one point
at which the density is highest.
It's the center of the distribution, the mean.
Now as you move away from the mean,
the density drops off, and it drops off
with this ellipsoidal contours.
So let's say the density of the mean is 0.5.
You can ask, "Let's look at all points
"whose density is 0.4.
"What are those points?"
Well, it might be this ellipse, for example.
And then you can say, "What are all points
"with density 0.3?"
Well, it might be this ellipse,
which is concentric with the first one.
And what are all points with density 0.2?
Well, it might be this ellipse, okay.
So what I'm showing over here are just contour lines
of this density.
The point of the highest density is the mean,
and then it falls off in these ellipsoidal contours, okay.
So let's look at these parameters again.
The mean makes a lot of sense.
It's just the center of the density.
What is this covariance matrix?
So this is a two-by-two matrix.
This number, over here, is the variance
along the x one direction.
It's just the variance of the alcohol value.
So we have 43 numbers, 43 alcohol levels.
We compute their variance, that's 0.2.
This number, over here, is the variance
along the second direction.
It's just the variance of all the flavonoid numbers.
The interesting thing is this number, over here.
This is the covariance between the two features,
between the alcohol level and the flavonoid level.
So let's quickly recall what that is.
What is the covariance between two random variables?
So let's say you have these two random variables,
x one and x two, alcohol and flavonoid, for example.
X one has mean mu one, average alcohol level.
X two has mean mu two, average flavonoid level.
A standard measure of the dependence
between x one and x two is the covariance.
It is the difference between the expected value
of x one times x two and the expected value
of x one times the expected value of x two.
When the covariance is positive,
it means there's a positive correlation.
When the covariance is negative,
it means there's a negative correlation, and if it's zero,
it means the two variables are uncorrelated.
So with this in mind, let's go and look
at the Gaussian distribution a little bit more formally.
Okay, so the two-dimensional Gaussian is a distribution
over 2D space.
It's a distribution over all of 2D space,
over all points x one, x two.
So to every point, it assigns some density.
Now the point of highest density is the mean,
mu one, mu two, so that's a 2D point.
Mu one is the average value of the x ones.
Mu two is the average value of the x twos.
That's the point of highest density.
The other parameters to this distribution
are variance and covariance parameters.
It's a two-by-two matrix
that contains four numbers.
But they're actually just three distinct numbers,
because it's a symmetric matrix
in which the two off-diagonal numbers are identical
to each other, okay.
So the number on the top left is the variance of x one.
The number in the bottom right is the variance of x two.
And the number that captures dependence,
there's just one number there, that's sigma one-two
or sigma two-one, and that's the covariance
of x one and x two, okay.
So these are the parameters of the 2D Gaussian distribution.
Now we already talked about the shape of this distribution.
The point of highest density is the mean mu.
And then the density falls off
with these ellipsoidal contours,
something like this,
where the shape of these ellipsoids is given
by the sigma matrix.
So now let's look at the precise form of the density.
So we've already discussed the parameters.
What is the formula for the density?
What is the density at any given point x one, x two?
Well, it has this formula over here,
which admittedly looks a little bit messy.
And there might be some terms that are confusing,
so let's just go through them one by one, okay.
So two, I think we can all agree on what that is.
Pi, this is just the pi 3.14.
It's that pi, okay.
What is this?
So sigma is a two-by-two matrix.
What does it mean to put two bars around sigma?
Well, that actually is the determinant of the matrix.
It's a way to write the determinant of sigma.
Some of you might be familiar with this, others not.
In either case, don't worry about it.
It's something that we'll be coming to again fairly soon.
The important thing to note
about this leading term over here
is that it does not depend on x at all.
It's just a normalizing constant.
It's just something that you have to stick in the front
to make the density sum up to one.
The really important stuff is happening
in here, inside the exponent.
And by the way, in case you haven't seen this before,
the notation, exp(z), is just another way
of writing e to the z, okay.
So the really important dependence arises from the exponent.
And let's look at those terms one by one, okay.
So in the middle, we have sigma inverse.
So sigma is a two-by-two matrix.
Its inverse is also a two-by-two matrix.
So this is a two-by-two matrix.
What's this thing over here,
x one minus mu one, x two minus mu two, okay?
Well, the center of the distribution is some point mu,
and we're trying to figure out the density
at some other point x.
So it turns out that the density depends only
on the displacement of x from mu.
It depends only on this displacement.
And that's what this vector is, over here.
It's a two-by-one vector.
It's just x minus mu.
It's the displacement of x from mu.
Mu is the center of the distribution.
X is some point whose density we're trying to figure out.
Now this is also x minus mu, but we've taken its transpose,
so it's now a row, so it's one-by-two.
And if this seems a little bit confusing,
these matrix inverses and transposes and so on,
again, you know, don't be concerned.
We will be going over these fairly soon.
But for the moment, we are multiplying together something
that's one-by-two with something that's two-by-two
with something that's two-by-one,
and so the answer is just one-by-one.
It's just a number, okay.
Now the smallest that number can be is zero,
and that's what happens when x is identically mu,
when you're just measuring the density at the center.
And that's when the density is highest.
As x moves away from mu, this number gets larger and larger,
and the density drops off.
So this is a little bit of insight
into the density of the bivariate Gaussian.
So let's look at a few examples,
just to solidify some of this, okay.
So what I've done here is to draw pictures
of two different Gaussians with the same mean.
So in both cases, the mean is the the point one one,
but they have different covariance matrices
and therefore different shapes.
So for each of them, I picked 50 points at random
from the Gaussian and plotted them.
Then I've also drawn a representative contour line.
Okay, so let's look at the one on the left for starters.
So the mean is one one.
So that's one one, and that is, of course,
the point right in the middle over there, okay.
The center of the distribution, that's the mean.
Okay, so that all makes sense.
Now let's look at the numbers in the covariance matrix.
Now the number at the upper left is the variance of x one,
which is four, so we know the standard deviation
of x one is the square root of the variance, so it's two.
And so the standard deviation of x two,
well, the variance of x two is one,
so the standard deviation is one, okay.
And the covariance between x one and x two is zero.
So what do these tell us?
So first of all, the standard deviation
of x one is twice that of x two.
So the data is twice as spread in the x one direction
as it is in the x two direction,
and we can see that directly from the plot, okay.
The other thing is that the covariance is zero.
So x one and x two are uncorrelated.
And again, we can see that, because the Gaussian, the data,
is not tilting either upwards or downwards.
It is axis-aligned, okay.
So now let's move on to the second picture.
So this is a Gaussian with exactly the same mean, one one.
The standard deviation of x one is, again, two.
The standard deviation of x two is, again, one.
So the stretch in the x one direction is, again,
twice the stretch in the x two direction.
But now, there's a covariance.
The covariance between x one and x two is 1.5,
which means that there is a bit
of a correlation between them, okay.
So x one and x two are correlated with each other.
Okay, so now we can go back to our winery example.
I think, hopefully, it's a little clearer now
what these ellipses actually mean.
What we did is we fitted a Gaussian
to winery number one, the red points,
and we'd just drawn a representative contour
of the Gaussian obtained in this way, okay.
And then we do the same for winery two and winery three.
And now, when a new point comes along,
we simply predict using Bayes' Rule.
Remember that we have class probabilities,
pi one, pi two, pi three.
When we get a new point x, we simply pick the class j
for which pi j times pj of x is maximized.
So what does that decision boundary look like?
It turns out this is what it is, okay.
Any point in here gets predicted
as being from winery number one.
Any point in here gets predicted
as being from winery number three, the green one.
And any point out here gets predicted
as being from winery number two.
It's rather a strange decision boundary, isn't it?
What is the functional form of this boundary?
Now that we've reduced the error to 8%,
by using two features, can we do better
by using more features?
How do we do that?
How do we take care of correlations
between multiple variables?
So we'll see the answers to all these questions very soon.
See you next time.

I included some posts for reference.

https://github.com/noey2020/How-to-Talk-Probability-Review-3

https://github.com/noey2020/How-to-Talk-Probability-Review-2

https://github.com/noey2020/How-to-Talk-Generative-Modeling-in-One-Dimension

https://github.com/noey2020/How-to-Talk-Probability-Review-1

https://github.com/noey2020/How-to-Talk-Generative-Approach-to-Classification

https://github.com/noey2020/How-to-Talk-of-Fitting-a-Distribution-to-Data-

https://github.com/noey2020/How-to-Talk-of-Host-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-of-Useful-Distance-Functions

https://github.com/noey2020/How-to-Talk-of-Improving-Nearest-Neighbor

https://github.com/noey2020/How-to-Talk-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-Matlab-Tricks-and-Tweaks

https://github.com/noey2020/How-to-Talk-Trading-and-Investing

https://github.com/noey2020/How-to-Work-in-Matlab-Development-Environment

https://github.com/noey2020/How-to-Talk-Vaccines

https://github.com/noey2020/How-to-Talk-Regression-in-Matlab

https://github.com/noey2020/How-to-Get-Started-in-Matlab

https://github.com/noey2020/How-to-Convert-Data-from-Web-Service-Using-Matlab

https://github.com/noey2020/Quote-for-the-Day

https://github.com/noey2020/How-to-Talk-Good-Investment-Strategy

https://github.com/noey2020/How-to-Talk-of-Good-Plan

https://github.com/noey2020/Thought-for-the-Day

https://github.com/noey2020/How-to-Talk-Stock-Watch-of-the-Day

https://github.com/noey2020/How-to-Talk-Data-Science

https://github.com/noey2020/How-to-Talk-Fundamental-Analysis

https://github.com/noey2020/How-to-Read-Company-Profiles

https://github.com/noey2020/How-to-Import-Data-from-Spreadsheets-and-Text-Files-Matlab-Without-Coding

https://github.com/noey2020/How-to-Talk-Model-of-Stock-Market-Prices-

https://github.com/noey2020/How-to-Talk-Digital-Wallets

https://github.com/noey2020/How-to-Talk-Investing

https://github.com/noey2020/How-to-Double-Your-Money-in-5years

https://github.com/noey2020/How-to-Talk-Matlab

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!
