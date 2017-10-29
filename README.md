# hansi :beers:
hansi :beers: counts bottles in a beer fridge and lets you know when there is too little cooled beer using opencv and a pretrained MobileNet SSD (Single shot multibox detector) network.

This was a fun hackerthon project and has absolutely no real world use case. (could be done way easier and more accurate by weighting the fridge for example)

### Hardware prerequisites
- a fridge with transparent door
- a camera pointing at the fridge (we used a gopro + hdmi capture card)
- some bottles in the fridge

### Software prerequisites
- [opencv-python](https://pypi.python.org/pypi/opencv-python)
- the caffemodel and prototext from https://github.com/chuanqi305/MobileNet-SSD#reproduce-the-result (pretrained MobileNet SSD network implemented in [caffe](https://github.com/BVLC/caffe) and trained with the [COCO dataset](http://cocodataset.org/#home). (this network can detect many object classes and is a complete overkill for detecting bottles)
- some notification service (we used rabbitmq and some in house backed slack service to notify a channel in case of cooled beer shortage emergencies)

### What it does
WIP

### How it works

**Segmenting the image**

 As we need to run two different algorithms to count the bottles in the fridge, the image of the fridge gets cut into an upper and lower part after getting the region of interest from the complete image.
 
 **Transforming the images**
 
The upper image part gets converted firstly into a gray image and some threshold filter is applied. The lower image part is not touched as the SSD was trained using regular rgb pictures.

**Running the algorithms**

On the upper image we use the Hough circle transformation to count the bottle capsules/corks. The bottles standing on the bottom of the fridge are counted by the SSD.

We found that the SSD count is quite stable in contrast to the Hough circle transformation so we developed a averaging algorithm to get the bottle count stable.

**Sending the notification**

Once the bottle count goes lower a configured limit a message is published to a rabbitmq queue. In our setup this triggers a slack service to post a message in the #beer channel.


### Contributors
- [@flofischer](https://github.com/flofischer)
- [@HalfbyteHeroes](https://github.com/HalfbyteHeroes)
- [@lukaskroepfl](https://github.com/lukaskroepfl)
- [@dominic-miglar](https://github.com/dominic-miglar)
