import torch

BATCH_SIZE = 8
RESIZE_TO = 300  # resize the image for training and transforms
NUM_EPOCHS = 10
NUM_WORKERS = 4

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DIR = '/home/giora/rip_current_detector/augmanted_training_data'
CLASSES = ['__background__', '1']
NUM_CLASSES = len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True

# location to save model and plots
OUT_DIR = 'outputs'
