# Pixel Objectness Demo
This repository contains Flask-based demo of foreground object segmentation approach named Pixel Objectness
https://github.com/suyogduttjain/pixelobjectness
## Instructions
Build Docker image:
```
cd docker
docker build -t pixelobjectness-demo .
```
Run Docker container:
```
# CPU-only version
docker run -e PORT=5000 -p 5000:5000 pixelobjectness-demo
# Or run on GPU 0
docker run -e GPU=0 -e PORT=5000 -p 5000:5000 pixelobjectness-demo
```
Process an image:
```
curl -F image=@image.jpg http://localhost:5000/predict -o output.jpg
```
Process an image and return mask:
```
curl -F image=@image.jpg -F mask=1 http://localhost:5000/predict -o output.jpg
```
Warning: image size must be not larger than 513x513.
