git pull
sudo docker build -t portal_machine_learning:latest .
sudo docker run -d -p 80:5000 portal_machine_learning