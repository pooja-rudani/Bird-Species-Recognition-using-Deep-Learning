import time
import json
from torch.utils.data import DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
import copy
import os

class Inceptionv3_Model:
    def __init__(self, path_to_pretrained_model=None, map_location='cpu', num_classes=10):
        """
        Allows for training, evaluation, and prediction of ResNet Models
        
        params
        ---------------
        path_to_pretrained_model - string - relative path to pretrained model - default None
        map_location - string - device to put model on - default cpu
        num_classes - int - number of classes to put on the deheaded ResNet
        """
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.feature_extract = True
        self.data_dir = "../dataset/"
        print("Initializing Datasets and Dataloaders...")
        # Create training and validation datasets
        self.data_transforms = self._setup_transform()
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x]) for x in ['train', 'valid', 'test']}
        # Create training and validation dataloaders
        self.dataloaders_dict = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=8, shuffle=True, num_workers=4) for x in ['train', 'valid', 'test']}
        if path_to_pretrained_model:
            self.model = torch.load(path_to_pretrained_model, map_location=map_location)
        else:
            self.model = self._setup_inceptionv3(num_classes=310)
            self.optimizer = self.create_optimizer(self.model)
            self.fit(self.model, self.dataloaders_dict, self.optimizer, num_epochs=5, is_inception=True)
            self.evaluate(self.model, self.dataloaders_dict)
            
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
            
    def _setup_transform(self):
        """
        Sets up transformations needed for train data, val data, and test data.
        Uses much of the image processing from ImageNet paper and includes some 
        image augmentation for training. Val and test transformers only perform
        minimum necessary processing.

        params
        ---------------
        None

        returns
        ---------------
        train_transform - torch transformer - transformer to use during training
        val_transform - torch transformer - transformer to use during validation
        test_transform - torch transformer - transformer to use during testing and inference
        """
        
        data_transforms = {
            'train': transforms.Compose([
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        return data_transforms
            
    def _setup_inceptionv3(self, num_classes):
        """
        Hidden function used in init if no pretrained model is specified. Helpful for implimenting transfer learning.
        It freezes all layers and then adds two final layers: one fully connected layer with RELU activation and dropout,
        and another as a final layer with number of class predictions as number of nodes. Also sends model to necessary device.

        params
        ---------------
        num_classes - int - Number of classes to predict

        returns
        ---------------
        model - torch model - torch model set up for transfer learning
        """
        
        model_ft = models.inception_v3(pretrained=True)
        self.set_parameter_requires_grad(model_ft, self.feature_extract)

        # Parameters of newly constructed modules have requires_grad=True by default
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, 310)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 310)
        return model_ft
    
    def create_optimizer(self, model):
        model = model.to(self.device)
        params_to_update = model.parameters()
        #print("Params to learn:")
        if self.feature_extract:
            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    #print("\t",name)
                else:
                    for name,param in model.named_parameters():
                        if param.requires_grad == True:
                            #print("\t",name)
                            pass
        
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        return optimizer_ft
        
    
    def fit(self, model, dataloaders, optimizer, criterion=None, num_epochs=5, is_inception=False):
        since = time.time()
        val_acc_history = []
        model = self.model
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        if not criterion:
            criterion = nn.CrossEntropyLoss()
        #if not scheduler:
        #    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                    
                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        # mode we calculate the loss by summing the final output and the auxiliary output
                        # but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            
                        _, preds = torch.max(outputs, 1)
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model, 'trained_model_inceptionv3.pt')
                if phase == 'valid':
                    val_acc_history.append(epoch_acc)
            print()
            
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
    
    def evaluate(self, model, dataloaders, criterion=None):
        """
        Feeds set of images through model and evaluates relevant metrics
        as well as batch predicts. Prints loss and accuracy

        params
        ---------------
        test_loader - torch DataLoader - Configured DataLoader for evaluation, helpful when images flow from directory
        model - trained torch model - Model to use during evaluation - default None which retrieves model from attributes
        criterion - Loss function to assess model - Default None which equates to CrossEntropyLoss.

        returns
        ---------------
        preds - list - List of predictions to use for evaluation of non-included metrics
        labels_list - list - List of labels to use for evaluation of non-included metrics
        """
        
        if not criterion:
            criterion = nn.CrossEntropyLoss()

        model.eval()
        test_loss = 0
        test_acc = 0
        preds = list()
        labels_list = list()

        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, pred = torch.max(outputs, dim=1)
                pred_a = pred.cpu().detach().numpy()
                preds.append(pred_a)
                labels_a = labels.cpu().detach().numpy()
                labels_list.append(labels_a)

            test_loss += loss.item() * inputs.size(0)
            correct = pred.eq(labels.data.view_as(pred))
            accuracy = torch.mean(correct.type(torch.FloatTensor))
            test_acc += accuracy.item() * inputs.size(0)

        test_loss = test_loss / len(dataloaders['test'].dataset)
        test_acc = test_acc / len(dataloaders['test'].dataset)

        print(f"Test loss: {test_loss:.4f}\nTest acc: {test_acc:.4f}")

        return preds, labels_list
    
    def predict_proba(self, img, k, index_to_class_labels, show=False):
        """
        Feeds single image through network and returns top k predicted labels and probabilities

        params
        ---------------
        img - PIL Image - Single image to feed through model
        k - int - Number of top predictions to return
        index_to_class_labels - dict - Dictionary to map indices to class labels
        show - bool - Whether or not to display the image before prediction - default False

        returns
        ---------------
        formatted_predictions - list - List of top k formatted predictions formatted to include a tuple of 1. predicted label, 2. predicted probability as str
        """
        if show:
            img.show()
        pred_transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = pred_transform(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        model =self.model
        model.eval()
        output_tensor = model(img)
        prob_tensor = torch.nn.Softmax(dim=1)(output_tensor)
        top_k = torch.topk(prob_tensor, k, dim=1)
        probabilites = top_k.values.detach().numpy().flatten()
        indices = top_k.indices.detach().numpy().flatten()
        formatted_predictions = []

        for pred_prob, pred_idx in zip(probabilites, indices):
            predicted_label = index_to_class_labels[pred_idx].title()
            predicted_prob = pred_prob * 100
            formatted_predictions.append((predicted_label, f"{predicted_prob:.3f}%"))

        return formatted_predictions
    
if __name__ == '__main__':
    #tests script to predict on single local image
    #model = Inceptionv3_Model(path_to_pretrained_model='trained_model_inceptionv3.pt')
    model = Inceptionv3_Model()
    #with open('index_to_class_label.json', 'rb') as f:
    #    j = json.load(f)
    #j = {int(k): v for k, v in j.items()}
    #img = Image.open('C:/Users/pjrud/Desktop/project/dataset/train/AFRICAN CROWNED CRANE/001.jpg')
    #print(model.predict_proba(img, 3, j, show=False))
    