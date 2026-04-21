from src.models.cnn import FinetunedDenseNet
from src.data.loader import *
import torch.nn as nn
from src.training import train_cnn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from src.training.unfreezer import ProgressiveUnfreezer
from sklearn.metrics import roc_auc_score



# aight lets just see if this works
def create_cnn_model(df, image_dir, device, mode, num_epochs=23):
    model = FinetunedDenseNet()
    model = model.to(device)
    # no longer need to do the weird he initialization, as I am using a pre-trained model. 

    # lets start with the CheXNet init params cause why not
    learning_rate = 0.0001 # decay by factor of 10 each time validation loss platueas after epoch
    # lets also make it 10x smaller, 1e-4 should work better. 


    # Adam optimizer, standard params. 
    train_loader, test_loader, pos_weight = get_dataloaders(df, image_dir, device) # after that I got a bunch of defaults, I'll change em when I get there.

    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # trying a different loss function, we shall see.
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    history, cnn_weights = train_model(model, train_loader, test_loader, num_epochs, device, pos_weight, mode=mode)
    return cnn_weights, history 




def train_model(model, train_loader, val_loader, num_epochs=23, device="cuda", pos_weight=0.5, mode="penumonia"):

    model = model.to(device)

    # lets add some early stopping stuff

    best_val_auc = 0.0
    paitence_counter = 0
    paitence = 5
    best_epoch = 0
    bset_model_state = None

    unfreezer = ProgressiveUnfreezer(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-3, 
        weight_decay=1e-4
    )

    # need to make sure this gets passed in. fantastic. 
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    torch.backends.cudnn.benchmark = True # apparently this is supposed to help
    

    best_val_auc = 0.0 # floor value
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(num_epochs):
        # first lets make sure we have all the appropriate layers that we want. 
        if unfreezer.should_update(epoch):
            optimizer = unfreezer.update(epoch, optimizer)

        # training time
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True) # non blocking for even more speed
            labels = labels.to(device, non_blocking=True) # ditto


            optimizer.zero_grad(set_to_none=True) # apparently more efficient than zero_Grad()
            outputs = model(images) # take that whole batch in 
            loss = loss_function(outputs.squeeze(), labels) # ouptuts need to be squeezed to change shape from (B, 1) to just (B)
            # backward and optimize        
            # pros and cons of keeping the clip grad norm? working keeping?    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # get loss value and clean up
            train_loss += loss.item()

            del outputs, loss, images, labels # make sure to clean these up to keep the vram nice and neat :)

        
        # now for the validatino phase (we run these back to back for finetuning, less epochs)
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = loss_function(outputs.squeeze(), labels)                
                val_loss += loss.item()

                preds = torch.sigmoid(outputs.squeeze())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # lets get rid of this 
                # print(f"Val positives: {sum(all_labels)} / {len(all_labels)}")
                # print(f"Prediction mean: {np.mean(all_preds):.4f}, std: {np.std(all_preds):.4f}")   

        val_auc = roc_auc_score(all_labels, all_preds)

        # early stopping logic goes here. 
        if val_auc > best_val_auc + 0.001:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0 # reset
            bset_model_state = model.state_dict().copy()
            # title = f"Best_densenet_{mode}.pth"
            # torch.save(model.state_dict(), title)
            print(f"Best new model! AUC {best_val_auc:.4f}")
        else:
            paitence_counter += 1
            print("Paitence did not improve")

        
        if paitence_counter >= paitence:
            print("Early stoopping at epoch ", epoch)
            model.load_state_dict(bset_model_state)
            break


        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # print(f"Epoch {epoch+1}/{num_epochs} | "
        #         f"Train Loss: {avg_train_loss:.4f} | "
        #         f"Val Loss: {avg_val_loss:.4f} | "
        #         f"Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            # torch.save(model.state_dict(), "Best_densenet_pneumonia.pth") # I want to save these manually, just in case. just trust me here. 
            # print(f"New best model! AUC {best_val_auc:.4f}")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_auc"].append(val_auc)

    return model.state_dict(), history # this should return the best model and the history thereof. 


