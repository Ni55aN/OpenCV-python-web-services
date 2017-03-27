sudo docker run -p 5000:5000 --name cv-services -d -v $(pwd):/src $1 bin/sh -c "cd /src && python app.py"
xdg-open http://localhost:5000/demo