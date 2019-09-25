## Install & use instructions

To run this Docker environment (and all the code in it), you only need to install
the latest versions of `Docker` and `docker-compose`. 

To start the Docker container (incl. serving Jupyter notebook at  `localhost:8888`),
run the following command in the top level of this repository:

`docker-compose up --build`

And to start the container shell:

`docker exec -it notebook bash`

See [the Docker documentation](https://docs.docker.com/) and [commonly used commands](https://towardsdatascience.com/15-docker-commands-you-should-know-970ea5203421).
