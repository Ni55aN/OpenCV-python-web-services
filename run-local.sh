sudo docker pull ni55an/python-opencv-modules
sudo docker run -p 5000:5000 --name cv-services -d -v $(pwd):/src ni55an/python-opencv-modules bin/sh -c "cd /src && python app.py"
xdg-open http://localhost:5000/demo