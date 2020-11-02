
**this project is largely dependent on richzhang paper and its pre-trained model**

this project uses deep learning to bring old tunisian images back to life. Making use of an already pretrained model I ceated a flask app that allows users to colorize their own black and white photos as well as explore our gallery.Then I managed to deploy this project on HEROKU ( link: imgcol.herokuapp.com )
**challenges**
-one of the challenges that I went  through during this project is dealing with output images from the model and then using them in the front this  issue is caused by image caching in the browser.
-Also since heroku is a saas I had to use custom buildpacks in order to add necessary dependcies to heroku's OS in order for OPENCV to work.

used links:
  *Black and white image colorization with OpenCV and Deep Learning: This article was very useful to inderstand the research paper  and to make use of the model loading it with OPENCV
https://www.pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/

@inproceedings{zhang2016colorful,
  title={Colorful Image Colorization},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A},
  booktitle={ECCV},
  year={2016}
}


