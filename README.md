# Multiclassification with CNNs on unbalanced dataset
## Introduction
This is the repository for the first challenge hosted by Politecnico di Milano for the Artificial neural networks and deep learning course in 2022.\
In this task, we are required to classify species of plants, which are
divided into 8 categories according to the species of the plant to which they belong. Being a classification
problem, given an image, the goal is to predict the correct class label.

## The dataset
The dataset provided is structured in a single folder containing the following classes:
- Species1 : 186 images
- Species2 : 532 images
- Species3 : 515 images
- Species4 : 511 images
- Species5 : 531 images
- Species6 : 222 images
- Species7 : 537 images
- Species8 : 508 images 


The dataset contains in total 3542 images of size 96x96.

### Examples of images

![00146](https://user-images.githubusercontent.com/62434812/204101912-fa4dae9c-49ac-4abe-b113-0fd52225473d.jpg)
![899800144](https://user-images.githubusercontent.com/62434812/204101929-0e9e3414-1054-4c6d-9a46-4068a20a92ed.jpg)
![Species2_00277](https://user-images.githubusercontent.com/62434812/204101943-8a07fd64-67c4-4852-82a7-988536e7b5e4.jpg)
![Species2_00307](https://user-images.githubusercontent.com/62434812/204101949-43b3dfd9-ac07-4ee4-a6d7-cfae08d4cbcf.jpg)
![Species2_00362](https://user-images.githubusercontent.com/62434812/204101969-837623c6-7c23-4f4d-ac99-6b7d0d8ad3b7.jpg)
![Species2_00391](https://user-images.githubusercontent.com/62434812/204101976-68afcfc5-4700-46bd-ad5b-f3b994f3170a.jpg)
![Species3_00113](https://user-images.githubusercontent.com/62434812/204101989-ac823aa2-64c9-4406-a518-af2a84fe3695.jpg)
![Species3_00194](https://user-images.githubusercontent.com/62434812/204101997-dbdbf9a9-1c71-4b9f-8a7e-b561496cfe37.jpg)

## Structure of the repository
The repository is structured in the following way:
- utilities: folder containing all the scripts used to manipulate the dataset before uploading it to kaggle.
- best_model.ipynb: notebook that generates the best model we were able to build.
- report.pdf: pdf file containing a brief recap of all our experiments before reaching the best model.

## Results
Our best model reached an accuracy of 91% on the validation set and 88.3% accuracy in the hidden test provided by the challenge host.\
F1-score table on the test set:
||Species1|Species2|Species3|Species4|Species5|Species6|Species7|Species8|
|---|---|---|---|---|---|---|---|---|
|F1-score|0.7704|0.8871|0.9292|0.8629|0.8980|0.9222|0.9632|0.8257|

Final position: 78 out of 300.
