"""
===============================================================================


Program Description
    A CNN that uses MNIST data set of handwritten letters to identify the numbers in the data.

    Author:         nunnas, nunnas@purdue.edu


Contributor:   N/A
    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.
    
ACADEMIC INTEGRITY STATEMENT
I have not used source code obtained from any other unauthorized
source, either modified or unmodified. Neither have I provided
access to my code to another. The project I am submitting
is my own original work.
===============================================================================
"""
#import statements 
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np 

from ENGR133_loaddata_nunnas import loaddata
def CNN (trainloader,testloader,classes):
#CNN
  class Net(nn.Module):
      def __init__(self): 
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
       #forward pass of the neural network  
      def forward(self, x): #x is a placeholder for the input of the data set (x is used so it can be implimented for other identifications)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*4*4) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
  net = Net()

#Loss function and optimizer
  criterion = nn.CrossEntropyLoss()
#customizing the learning rate (.001 works well for MNIST data set)
  while True:
    try:
        while True:
            lr_str = input("Enter the learning rate: ")
            lr = float(lr_str)
            if lr > 0:
                break  # Exit the inner loop if a valid positive float is entered
            else:
                print("Error: Learning rate should be a positive number.")
    except ValueError:
        print("Error: Please enter a valid number for the learning rate.")
    else:
        break  # Exit the outer loop if a valid input is received.01

    # Set up the optimizer with the user-customized learning rate
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

  #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#training network
  while True:
    epoch_str = input("Enter the number of epochs: ")
    try:
        epoch = int(epoch_str)
        if epoch > 0:
            break  # Exit the loop if a valid positive integer is entered
        else:
            print("Error: Number of epochs should be a positive integer.")
    except ValueError:
        print("Error: Please enter a valid integer for the number of epochs.")

  for epoch in range(1): # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
#get the inputs; data is a list of [inputs, labels]
      inputs, labels = data
 # zero the parameter gradients
      optimizer.zero_grad()
 # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
 # print statistics
      running_loss += loss.item()
      if i % 2000 == 1999: # print every 2000 mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0


  print('Finished Training')

#saving path 
  PATH = './MNIST_net.pth'
  torch.save(net.state_dict(), PATH)
#trying to show the images but the plotting size is not matching up 

  dataiter = iter(testloader)
  images, labels = next(dataiter)
# If the images have more than one channel, permute and squeeze
  if images.shape[1] > 1:
    images = images.permute(0, 2, 3, 1) # Change channel dimension order
    images = images.squeeze()

# Print images
  grid = torchvision.utils.make_grid(images)

# If the grid has more than 3 channels, select only the first 3
  if grid.shape[0] > 3:
    grid = grid[:3]

# Transpose the grid for plotting
  grid = grid.permute(1, 2, 0)

  print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

  net = Net()
  net.load_state_dict(torch.load(PATH))
  outputs = net(images)
  _, predicted = torch.max(outputs, 1)
  print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
  for j in range(4)))
  correct = 0
  total = 0
  with torch.no_grad():
    for data in testloader:
       images, labels = data
       outputs = net(images)
       _, predicted = torch.max(outputs.data, 1)
       total += labels.size(0)
       correct += (predicted == labels).sum().item()  
  print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# accuracy for each class 
  correct_pred = {classname: 0 for classname in classes}
  total_pred = {classname: 0 for classname in classes}

# again no gradients needed
  with torch.no_grad():
      for data in testloader:
          images, labels = data
          outputs = net(images)
          _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
          for label, prediction in zip(labels, predictions):
              if label == prediction:
                  correct_pred[classes[label]] += 1
              total_pred[classes[label]] += 1

# print accuracy for each class
  for classname, correct_count in correct_pred.items():
      accuracy = 100 * float(correct_count) / total_pred[classname]
      print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
 
  plt.title("Sample images")
  plt.imshow(grid,cmap='gray')
  plt.axis('off')
  plt.show() # Show the plot
  return 
if __name__ == '__main__':
    # Load data using the loaddata function from ENGR133_loaddata_nunnas.py
    trainloader, testloader, classes = loaddata()
    # Call the CNN function with loaded data
    CNN(trainloader, testloader, classes)
