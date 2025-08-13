Abstract

Using three-dimensional geometric features acquired from a video using a smartphone.
This study builds a method that is novel, non-destructive and cost-effective for fore-
casting the nutritional content of fruit. A consumer-grade smartphone is used in this
methodology to record multi-view videos of eight different types of fruits. These videos
are then processed into three-dimensional models using Structure-from-Motion (SfM)
in Meshroom, and key geometric features such as volume, surface area, sphericity,
and aspect ratio are extracted from the models. Through the use of a random forest
regression model, the USDA FoodData Central database’s nutritional information (cal-
ories, carbs, fibre, sugar, protein, fat, and vitamin C) is obtained with these factors.
With R2 values of 0.69 for calories and 0.77 for vitamin C, the results show that the
predictive accuracy is moderate. However, the Mean Absolute Errors indicate that there
are limitations in precision. The research results demonstrate that there is a strong
linear correlation between volume and weight, as R2 equals 1 under the assumption
of equal density. However, it also underlines the difficulties associated with scaling
accuracy and the diversity of datasets. A proof of concept for accessible nutritional
evaluation is established by this prototype. It has benefits in precision agriculture and
dietary monitoring. And also, future study suggests that larger datasets, improved
reconstruction techniques, and the incorporation of spectral data be utilised in order to
improve the reliability of the assessment.
