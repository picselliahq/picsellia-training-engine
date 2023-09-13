<h1 align="center">
    <img margin="0 10px 0 0" src="https://www.docker.com/wp-content/uploads/2022/03/vertical-logo-monochromatic.png" width="150px"/>
    <img margin="0 10px 0 0" src="https://uploads-ssl.webflow.com/60d1a7f5aeb33cb8af546898/610bcf4cc7ae73979fa0d23b_256.png" width="120px"/>
</h1>
  <h2 align="center">Picsellia Public Repo</h2>
  <p align="center">A collection of Docker images provided for your trainings!
<p>
<p align="center">
    <a href="https://www.python.org/downloads/" target="_blank"><img src="https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10-brightgreen.svg" alt="Python supported"/></a>
</p>

The Picsellia Training Engine is a project designed to provide an efficient, customizable, and Docker-based system for
training ML models. Its main purpose is to interact seamlessly with the Picsellia SAS platform,
leveraging the functionalities and capabilities it provides.

The repository provides images for the following ML Frameworks:

- Tensorflow 2
- Keras
- ONNX

# Installation ⚙️

You'll just need to install [Docker](https://docs.docker.com/engine/install/), then you can:

- Select an image from this repository and modify it to suit your needs. Please ensure you follow
  our [specific guidelines](https://dash.readme.com/project/picsellia-docs/v2.2/docs/integrate-picsellia-into-your-training-scripts)
  to ensure compatibility with Picsellia.
- Create your own Dockerfile and training script. For this, you can use one of our base images. Pull it directly from
  Docker Hub using `picsellia/cuda:<base-tag>`. The available tags
  are [here](https://hub.docker.com/r/picsellia/cuda/tags).

# Configuration 🛠️️

If you create your own Dockerfile, its structure should look like this:

```Dockerfile
FROM picsellia/cuda:<base-tag>

COPY your-custom-image/requirements.txt .
RUN pip install -r requirements. txt

# Note: You can also put the experiment package inside your requirements
RUN pip install -r picsellia --upgrade

WORKDIR /picsellia

COPY your-custom-image/picsellia .

ENTRYPOINT ["run", "main.py"] 
```

Using `ENTRYPOINT ["run", "main.py"]` will ensure that the log container's output is automatically directed to your
Picsellia workspace.

# Contributing 🤝

We welcome contributions from the community! While we're still working on establishing a full set of guidelines, we
encourage you to adhere to the general principles of respect for the original project, clarity in any changes you make,
and supporting explanations for your contributions.

# License 📄

This project is licensed under the MIT License. For more information, please refer to the LICENSE file in the
repository.

# Contact Information 🥑

Should you have any questions or if you want to contribute, please don't hesitate to contact us.

You can reach us:

- On our website https://picsellia.com/contact.
- By email at: [support@picsellia.com](mailto:support@picsellia.com).
- On Github: https://github.com/PN-picsell ☕
