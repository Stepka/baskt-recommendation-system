FROM ubuntu

# Update
RUN apt-get update
RUN apt-get install -y python 
RUN apt-get install -y python-pip
 
# Install app dependencies
RUN pip install --upgrade pip

# Install app dependencies
RUN pip install --upgrade Flask
RUN pip install --upgrade --user ortools
RUN pip install googlemaps

# Bundle app source
COPY Final_route.py Final_route.py

EXPOSE  8000
CMD ["python", "Final_route.py", "-p 8000"]