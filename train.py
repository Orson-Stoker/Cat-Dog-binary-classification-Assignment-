from resnet import *
from dnn import *
from rnn import *
from cnn import *
from dataloader import *


def train(model,train_loader,test_loader,learning_rate=0.01,epoch=10,gpu=True):
    total_train_step=0  
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
    
    if gpu==True:
        model=model.cuda()
        loss_fn=loss_fn.cuda()


    test_loss=[]
    test_acc=[]
    for i in range(epoch):
        print("------------ training turn {}-------------".format(i+1))
        model.train()
        for data in train_loader:
            imgs,targets=data
            if gpu==True:
                imgs=imgs.cuda()
                targets=targets.cuda()

            outputs=model(imgs)
            loss= loss_fn(outputs,targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_step=total_train_step+1
            if total_train_step%100 ==0:
                print(f"training steps:{total_train_step}, loss:{loss}")

        total_test_loss=0
        total_accuracy=0
        test_datasize=500


        model.eval()
        with torch.no_grad():
            for data in test_loader:
                imgs,targets = data
                if gpu==True:
                    imgs=imgs.cuda()
                    targets=targets.cuda()
                    
                outputs=model(imgs)
                loss = loss_fn(outputs,targets)
                accuracy=(outputs.argmax(1)==targets).sum()

                total_test_loss=total_test_loss+ loss
                total_accuracy=total_accuracy +accuracy
        test_loss.append(total_test_loss)
        test_acc.append(round(float(total_accuracy/test_datasize),5))
        print(f"loss in test dataset : {total_test_loss}")
        print(f"accuracy in test dataset : {total_accuracy/test_datasize}")
    print("--------------result----------------")
    print(test_acc)
               

if __name__ == "__main__":
    model=Resnet()
    train(model,train_loader,val_loader)
    

        