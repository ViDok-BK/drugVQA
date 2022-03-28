from torch._C import device
from model import *
from dataPre import *
from sklearn import metrics
import pickle
import torch

device = torch.device('cpu')
model = testArgs['model']
model.load_state_dict(torch.load(model_path, map_location="cuda:0"))

def getROCE(predList,targetList,roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index,x] for index,x in enumerate(predList)]
    predList = sorted(predList,key = lambda x:x[1],reverse = True)
    tp1 = 0
    fp1 = 0
    maxIndexs = []
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate*n)/100)):
                break
    roce = (tp1*n)/(p*fp1)
    return roce

def testPerProtein(testArgs, model):
    result = {}
    for x in testArgs['test_proteins']:
        print('\n current test protein:',x.split('_')[0])
        data = testArgs['testDataDict'][x]
        test_dataset = ProDataset(dataSet = data,seqContactDict = testArgs['seqContactDict'])
        test_loader = DataLoader(dataset=test_dataset,batch_size=1, shuffle=True,drop_last = True)
        testArgs['test_loader'] = test_loader
        testAcc,testRecall,testPrecision,testAuc,testLoss,all_pred,all_target,roce1,roce2,roce3,roce4 = test(testArgs, model)
        result[x] = [testAcc,testRecall,testPrecision,testAuc,testLoss,all_pred,all_target,roce1,roce2,roce3,roce4]
    return result

def test(testArgs, model):
    test_loader = testArgs['test_loader']
    criterion = testArgs["criterion"]
    C = testArgs['penal_coeff']
    attention_model = model
    losses = []
    accuracy = []
    print('test begin ...')
    total_loss = 0
    n_batches = 0
    correct = 0
    all_pred = np.array([])
    all_target = np.array([])
    with torch.no_grad():
        for batch_idx,(lines, contactmap,properties) in enumerate(test_loader):
            input, seq_lengths, y = make_variables(lines, properties,smiles_letters)
            attention_model.hidden_state = attention_model.init_hidden()
            contactmap = contactmap.cpu()
            y_pred,att = attention_model(input,contactmap)
            if trainArgs['use_regularizer']:
                attT = att.transpose(1,2)
                identity = torch.eye(att.size(1))
                identity = Variable(identity.unsqueeze(0).expand(train_loader.batch_size,att.size(1),att.size(1))).cuda()
                penal = attention_model.l2_matrix_norm(att@attT - identity)
            if not bool(attention_model.type) :
                #binary classification
                #Adding a very small value to prevent BCELoss from outputting NaN's
                pred = torch.round(y_pred.type(torch.DoubleTensor).squeeze(1))
                correct+=torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),y.type(torch.DoubleTensor)).data.sum()
                all_pred=np.concatenate((all_pred,y_pred.data.cpu().squeeze(1).numpy()),axis = 0)
                all_target = np.concatenate((all_target,y.data.cpu().numpy()),axis = 0)
                if trainArgs['use_regularizer']:
                    loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))+(C * penal.cpu()/train_loader.batch_size)
                else:
                    loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))
            total_loss+=loss.data
            n_batches+=1
    testSize = round(len(test_loader.dataset),3)
    testAcc = round(correct.numpy()/(n_batches*test_loader.batch_size),3)
    testRecall = round(metrics.recall_score(all_target,np.round(all_pred)),3)
    testPrecision = round(metrics.precision_score(all_target,np.round(all_pred)),3)
    testAuc = round(metrics.roc_auc_score(all_target, all_pred),3)
    print("AUPR = ",metrics.average_precision_score(all_target, all_pred))
    testLoss = round(total_loss.item()/n_batches,5)
    print("test size =",testSize,"  test acc =",testAcc,"  test recall =",testRecall,"  test precision =",testPrecision,"  test auc =",testAuc,"  test loss = ",testLoss)
    roce1 = round(getROCE(all_pred,all_target,0.5),2)
    roce2 = round(getROCE(all_pred,all_target,1),2)
    roce3 = round(getROCE(all_pred,all_target,2),2)
    roce4 = round(getROCE(all_pred,all_target,5),2)
    print("roce0.5 =",roce1,"  roce1.0 =",roce2,"  roce2.0 =",roce3,"  roce5.0 =",roce4)
    return testAcc,testRecall,testPrecision,testAuc,testLoss,all_pred,all_target,roce1,roce2,roce3,roce4

testPerProtein(testArgs, model)
