import requests

# You can use your own uploaded images and captions. 
# You will be responsible for the legal use of images that 
#  you are going to use.

url1='http://farm3.staticflickr.com/2519/4126738647_cc436c111b_z.jpg'
cap1='A motorcycle sits parked across from a herd of livestock'

url2='http://farm3.staticflickr.com/2046/2003879022_1b4b466d1d_z.jpg'
cap2='Motorcycle on platform to be worked on in garage'

url3='http://farm1.staticflickr.com/133/356148800_9bf03b6116_z.jpg'
cap3='a cat laying down stretched out near a laptop'

img1 = {
  'flickr_url': url1,
  'caption': cap1,
  'image_path' : './data/motorcycle_1.jpg'
}

img2 = {
    'flickr_url': url2,
    'caption': cap2,
    'image_path' : './data/motorcycle_2.jpg'
}

img3 = {
    'flickr_url' : url3,
    'caption': cap3,
    'image_path' : './data/cat_1.jpg'
}

# download images
imgs = [img1, img2, img3]
for img in imgs:
    data = requests.get(img['flickr_url']).content
    with open(img['image_path'], 'wb') as f:
        f.write(data)