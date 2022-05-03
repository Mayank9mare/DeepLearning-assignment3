import matplotlib.pyplot as plt
file1 = open('search-EXP-20220501-011924/log.txt', 'r')




tval = {'train_acc':[],'val_acc':[],'x':[],'y':[]}
Lines = file1.readlines()
for line in Lines:
    #line=line.strip()
    l=line.split(' ')
    for x in range(len(l)):
        if(l[x]=='train_acc'):
            tval['train_acc'].append(100-float(l[x+1]))
        if(l[x]=='valid_acc'):
            tval['val_acc'].append(100-float(l[x+1]))





def loss_curve(tval):
    plt.figure(figsize=(5,4))
    # plt.plot(list(range(1,len(tval['val_acc'])+1)),tval['val_acc'],label='validation accuracy')
    # plt.plot(list(range(1,len(tval['train_acc'])+1)),tval['train_acc'],label='training accuracy')
    plt.plot(list(range(1,len(tval['train_loss'])+1)),tval['train_loss'],label='training loss')
    plt.plot(list(range(1,len(tval['val_loss'])+1)),tval['val_loss'],label='validation loss')

    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.legend()
    plt.savefig('loss5x1')

def loss_curve2(tval):
    plt.figure(figsize=(5,4))
    plt.plot(list(range(1,len(tval['val_acc'])+1)),tval['val_acc'],label='validation error')
    plt.plot(list(range(1,len(tval['train_acc'])+1)),tval['train_acc'],label='training error')
    # plt.plot(list(range(1,len(tval['train_loss'])+1)),tval['train_loss'],label='training loss')
    # plt.plot(list(range(1,len(tval['val_loss'])+1)),tval['val_loss'],label='validation loss')

    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.title('accuracy curve')
    plt.legend()
    plt.savefig('error')
loss_curve2(tval)
