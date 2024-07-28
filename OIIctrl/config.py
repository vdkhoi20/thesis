import torch
class Config:
    SEED=42

    INITIAL_ALPHA=0.23
    
    al=0.4
    k=5
    
    SCALE_MASK=0.1
    THRES_HOLD=0.25

    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FOREGROUND_MASK=True
    USE_INTERPOLIATE=True
    USE_INTERPOLIATE_INTER=True
    HEIGHT=512
    WIDTH=512
    MAX_STEP=50
    GUIDANCE_SCALE=7.5
    STEP_QUERY=7
    LAYER_QUERY=15
    STEP_CHANGE_MASK=1

class ConfigExpandMask(Config):
    SCALE_MASK=0.17
class ConfigSimilarObject(Config):
    STEP_QUERY=3
    LAYER_QUERY=7
    SCALE_MASK=0.01
class ConfigFixedMask(Config):
    SCALE_MASK=0.002
    
