## DeepCrysTet: A Deep Learning Approach Using Tetrahedral Mesh for Predicting Properties of Crystalline Materials

DeepCrysTet is a novel deep learning approach for predicting material properties, which uses crystal structures represented as a 3D tetrahedral mesh generated by Delaunay tetrahedralization.
DeepCrysTet provides a useful framework comprising three core components: a 3D mesh generation method, mesh-based feature design, and neural network design.
The overall framework of DeepCrysTet is shown below.

<img src="./docs/images/model_architecture.png" alt="model-architecture">

## Table of Contents

- [Dataset](#dataset)
- [Training DeepCrysTet](#training-deepcrystet)
- [Citation](#citation)

## Datasets

The evaluation dataset used in the original DeepCrysTet paper is generated from the 2018.10.18 version of the Materials Project dataset.
You can download the dataset below.

| Dataset                                |                                        Download                                        |
|----------------------------------------|:--------------------------------------------------------------------------------------:|
| Materials Project (2018.10.18 version) |      [Link](https://figshare.com/articles/dataset/Materials_Project_Data/7227749)      |
| DeepCrysTet's Supervised Data          | [Link](https://figshare.com/articles/dataset/3D_Mesh_Dataset_for_DeepCrysTet/22031969) |

If you want to learn more about the data generation process or create your own 3D mesh dataset, more information can be found in the [data](data/) folder.

## Training DeepCrysTet

### Environment

We use [Poetry](https://python-poetry.org/) for managing our packages.
To get started, clone `DeepCrysTet` repository and run the following command from the root directory of this repository.

```bash
poetry install --no-root
```

Run the following command to activate the environment:

```bash
poetry shell
```

### Run Training

The model is trained using [DeepCrysTet's supervised data](https://figshare.com/articles/dataset/3D_Mesh_Dataset_for_DeepCrysTet/22031969) by executing the following commands.
## Citation

TBD