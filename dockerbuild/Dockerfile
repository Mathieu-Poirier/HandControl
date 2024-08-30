# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /CNNHOST

# Copy the current directory contents into the container at /app
COPY . /CNNHOST

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r server_requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=server.py

# Run server.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]