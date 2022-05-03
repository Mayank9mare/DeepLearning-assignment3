import matplotlib.pyplot as plt
import os

direct = "search-EXP-20220501-011924/log.txt"

lr_vs_epochs = {'lr':[],'epochs':[]}
val_loss = {'loss':[],'steps':[]}
val_acc = {'steps':[],'top1':[],'top5':[]}
train_loss = {'loss':[],'steps':[]}
train_acc = {'steps':[],'top1':[],'top5':[]}

with open(direct,'r') as f:
    for line in f.readlines():
        line = line.replace('\n','')
        if 'epoch' in line and 'weight' not in line:
            line = line.split()
            lr = float(line[-1])
            epoch = int(line[-3])
            if epoch == 20:
                break
            lr_vs_epochs['lr'].append(lr)
            lr_vs_epochs['epochs'].append(epoch)
        elif 'train' in line and 'train_acc' not in line and 'weight' not in line:
            line = line.split()
            loss = float(line[-3])
            steps = int(line[-4])
            top1 = float(line[-2])
            top5 = float(line[-1])
            train_loss['loss'].append(loss)
            train_loss['steps'].append(steps)
            train_acc['steps'].append(steps)
            train_acc['top1'].append(top1)
            train_acc['top5'].append(top5)
        elif 'valid' in line and 'valid_acc' not in line and 'weight' not in line:
            line = line.split()
            loss = float(line[-3])
            steps = int(line[-4])
            top1 = float(line[-2])
            top5 = float(line[-1])
            val_loss['loss'].append(loss)
            val_loss['steps'].append(steps)
            val_acc['steps'].append(steps)
            val_acc['top1'].append(top1)
            val_acc['top5'].append(top5)

# lr_vs_epochs = {'lr':[],'epochs':[]}
# val_loss = {'loss':[],'steps':[]}
# val_acc = {'steps':[],'top1':[],'top5':[]}
# train_loss = {'loss':[],'steps':[]}
# train_acc = {'steps':[],'top1':[],'top5':[]}
plt.figure(figsize=(5,4))
plt.plot(lr_vs_epochs['epochs'],lr_vs_epochs['lr'])
plt.xlabel('epochs')
plt.ylabel('lr')
plt.title('lr vs epochs')
plt.savefig('lve.svg',format='svg')
plt.show()

plt.figure(figsize=(5,4))
plt.plot(val_loss['steps'],val_loss['loss'])
plt.xlabel('steps')
plt.ylabel('loss')
plt.title('val loss vs steps')
plt.savefig('vlvs.svg',format='svg')
plt.show()

plt.figure(figsize=(5,4))
plt.plot(train_loss['steps'],train_loss['loss'])
plt.xlabel('steps')
plt.ylabel('loss')
plt.title('train loss vs steps')
plt.savefig('tlvs.svg',format='svg')
plt.show()

plt.figure(figsize=(5,4))
plt.plot(train_acc['steps'],train_acc['top1'],label='top1')
plt.plot(train_acc['steps'],train_acc['top5'],label='top5')
plt.xlabel('steps')
plt.ylabel('accuracy')
plt.title('train accuracy vs steps')
plt.legend()
plt.savefig('tavs.svg',format='svg')
plt.show()

plt.figure(figsize=(5,4))
plt.plot(val_acc['steps'],val_acc['top1'],label='top1')
plt.plot(val_acc['steps'],val_acc['top5'],label='top5')
plt.xlabel('steps')
plt.ylabel('accuracy')
plt.title('val accuracy vs steps')
plt.legend()
plt.savefig('vavs.svg',format='svg')
plt.show()
