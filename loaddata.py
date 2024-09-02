#side script 
"""
===============================================================================


Program Description
    A CNN that uses MNIST data set of handwritten letters to identify the numbers in the data.
    (side functions loading in the data)

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
import torch
import torchvision
import torchvision.transforms as transforms

#Loading in data set by transforming and seperating classes of definition 
def loaddata():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    batch_size = 4
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
    download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    shuffle=True, num_workers=0)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
    download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    shuffle=False, num_workers=0)
    classes = ('0', '1', '2', '3',
    '4', '5', '6', '7', '8', '9')
    return trainloader,testloader,classes
