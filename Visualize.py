import matplotlib.pyplot as plt

##add on after defining Dataloader and Model

def visualize(predicted,target,inputs):
    outputs = F.softmax(output, dim=1)
    predicted = outputs.max(1, keepdim=True)[1]

    fig, axs = plt.subplots(4,8,figsize=(15,7))
    axs = axs.reshape(-1)
    fig.suptitle('Prediction')
    for i in range(32):
        img_arr = predicted[0,0,i,:,:]
        axs[i].imshow(img_arr, cmap='hot')
        axs[i].set_title('%d'%i)
        axs[i].set_axis_off()

    fig, axs = plt.subplots(4,8,figsize=(15,7))
    axs = axs.reshape(-1)
    fig.suptitle('Targets')
    for i in range(32):
        img_arr = target[0,i,:,:]
        axs[i].imshow(img_arr, cmap='hot')
        axs[i].set_title('%d'%i)
        axs[i].set_axis_off()

    fig, axs = plt.subplots(4,8,figsize=(15,7))
    axs = axs.reshape(-1)
    fig.suptitle('Inputs')
    for i in range(32):
        img_arr = inputs[0,0,i,:,:]
        axs[i].imshow(img_arr, cmap='hot')
        axs[i].set_title('%d'%i)
        axs[i].set_axis_off()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#test_data=[(736, 256, 288), (800, 704, 288), (480, 480, 544), (608, 736, 736), (480, 928, 32)]
test_data=[(512, 64, 800), (480, 192, 96), (704, 96, 896), (384, 672, 448), (64, 640, 384)]
# #build dataloader
params = {'batch_size': 1,
      'shuffle': False,
      'num_workers':20}
testing_set= Dataset(test_data)
testing_generator = data.DataLoader(testing_set, **params)

############Change this part for different model######
model=Modified3DUNet(1,2).to(device)
state_dict = torch.load('Test2_1030/checkpoint_1030_all.pth')
model.load_state_dict(state_dict)
#######################################################

model.eval()
with torch.no_grad():
    for i, (inputs, target) in enumerate(testing_generator):
        inputs = inputs.unsqueeze(dim = 1).to(device).float()
        target = target.to(device).long()            
        output = model(inputs)
        outputs = F.softmax(output, dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        dm_mean=5.614541471004486
        visualize(predicted,target,inputs) #*dm_mean+dm_mean)
