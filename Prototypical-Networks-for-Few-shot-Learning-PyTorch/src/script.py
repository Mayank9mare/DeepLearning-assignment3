import matplotlib.pyplot as plt
file1 = open('../output5x1/val_acc.txt', 'r')
file2 = open('../output5x1/train_acc.txt', 'r')
file3 = open('../output5x1/val_loss.txt', 'r')
file4 = open('../output5x1/train_loss.txt', 'r')



tval = {'train_acc':[],'val_acc':[],'train_loss':[],'val_loss':[]}
Lines = file1.readlines()
for line in Lines:
    line=line.strip()
    a=float(line)
    tval['val_acc'].append(a)

Lines = file2.readlines()
for line in Lines:
    line=line.strip()
    a=float(line)
    tval['train_acc'].append(a)
Lines = file3.readlines()
for line in Lines:
    line=line.strip()
    a=float(line)
    tval['val_loss'].append(a)
Lines = file4.readlines()
for line in Lines:
    line=line.strip()
    a=float(line)
    tval['train_loss'].append(a)

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
    plt.plot(list(range(1,len(tval['val_acc'])+1)),tval['val_acc'],label='validation accuracy')
    plt.plot(list(range(1,len(tval['train_acc'])+1)),tval['train_acc'],label='training accuracy')
    # plt.plot(list(range(1,len(tval['train_loss'])+1)),tval['train_loss'],label='training loss')
    # plt.plot(list(range(1,len(tval['val_loss'])+1)),tval['val_loss'],label='validation loss')

    plt.xlabel('iterations')
    plt.ylabel('acc')
    plt.title('accuracy curve')
    plt.legend()
    plt.savefig('acc5x1')
loss_curve(tval)
loss_curve2(tval)