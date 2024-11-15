from scene.my_gaussian_model import GaussianModel

class Config:
    optimizer_type = 'default'
    sh_degree=3
gs = GaussianModel(Config())