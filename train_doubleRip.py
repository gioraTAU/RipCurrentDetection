from config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, NUM_WORKERS, BATCH_SIZE
)
from model import create_model
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt
import time
import pandas as pd
from torch.utils.data import DataLoader
from costumDataset import DoubleRipCurrentDataset
import torchvision.transforms as T


plt.style.use('ggplot')


# function for running training iterations
def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        d = []
        for iii, jjj in zip(targets['labels'], targets['box']):
            #cur_element = {'labels': iii.to(DEVICE).unsqueeze(0), 'boxes': jjj.unsqueeze(0).to(DEVICE)}
            cur_element = {'labels': iii.to(DEVICE), 'boxes': jjj.to(DEVICE)}

            d.append(cur_element)

        #targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        #targets = [targets['box'], targets['labels']]
        loss_dict = model(images, d)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list


# function for running validation iterations
def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        d = []
        for iii, jjj in zip(targets['labels'], targets['box']):
            #cur_element = {'labels': iii.to(DEVICE).unsqueeze(0), 'boxes': jjj.unsqueeze(0).to(DEVICE)}
            cur_element = {'labels': iii.to(DEVICE), 'boxes': jjj.to(DEVICE)}

            d.append(cur_element)
            #cur_element = {'labels': iii.to(DEVICE).unsqueeze(0), 'boxes': jjj.unsqueeze(0).to(DEVICE)}
            #d.append(cur_element)

        with torch.no_grad():
            loss_dict = model(images, d)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list


if __name__ == '__main__':

    df = pd.read_csv('/home/giora/rip_current_detector/doubleRip_aug_data_labels.csv')
    df_randomized = df.sample(len(df))
    df_train = df_randomized[0:int(0.8*len(df))]
    df_val = df_randomized[int(0.8*len(df)):]
    img_dir = '/home/giora/rip_current_detector/training_data_two_rips/'
    trans = T.ToTensor()
    train_dataset = DoubleRipCurrentDataset(df_train, img_dir, trans)
    valid_dataset = DoubleRipCurrentDataset(df_val, img_dir, trans)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")
    # initialize the model and move to the computation device
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []
    # name to save the trained model with
    MODEL_NAME = 'model'
    # whether to show transformed images from data loader or not
    if VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_tranformed_image

        show_tranformed_image(train_loader)
    # initialize SaveBestModel class
    save_best_model = SaveBestModel(OUT_DIR)
    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {NUM_EPOCHS}")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()
        # start timer and carry out training and validation
        start = time.time()
        model.train()
        train_loss = train(train_loader, model)
        #model.eval()
        val_loss = validate(valid_loader, model)
        print(f"Epoch #{epoch + 1} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch + 1} validation loss: {val_loss_hist.value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        # save the best model till now if we have the least loss in the...
        # ... current epoch
        save_best_model(
            val_loss_hist.value, epoch, model, optimizer
        )
        # save the current epoch model
        save_model(OUT_DIR, epoch, model, optimizer)
        # save loss plot
        save_loss_plot(OUT_DIR, train_loss, val_loss)

        # sleep for 5 seconds after each epoch
        time.sleep(5)