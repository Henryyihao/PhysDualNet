from .PhysDualNet import PhysDualNet

model_dict = {
    "PhysDualNet": PhysDualNet,
}


def get_model_dict():
    return model_dict

