#coding=utf8
from neuralnetworktool import *

import json
import copy
import random
import traceback

class Neuron_output():
    def __init__(self,nHide = 0,studySpeed = 1,thresholdFunc = sigmoidX):
        self.nHide = nHide
        #使用随机数初始化权值
        randomWeights = [random.uniform(-1, 1) for i in range(nHide)]
        randomThreshold = random.uniform(0, 1)
        self.weights = randomWeights
        self.threshold = randomThreshold
        self.thresholdFunc = thresholdFunc
        self.studyspeed = studySpeed
        self.inputs = []
    
    def setInputs(self,inputs):
	    self.inputs = inputs

    def getInputs(self):
	    return self.inputs

    def getOutput(self,inputs):
        self.inputs = inputs
        output = np.squeeze(np.asarray(self.weights * self.inputs))
        return round(self.thresholdFunc(output-self.threshold),3)
    
    def getWeights(self):
	    return np.squeeze(np.asarray(self.weights))

    def getWeightsMatrix(self):
	    return self.weights

    def setWeights(self,weights):
	    self.weights = np.matrix(weights)

    def setStudySpeed(self,studyspeed):
        self.studyspeed = studyspeed

    def getR(self,inputs,exceptOutput,realOutput = None):
        if realOutput == None:
            f = self.getOutput(inputs)
        else:
            f = realOutput
        d = exceptOutput
        R = f * (1 - f) * (f - d)
        self.adjustWeightsWithR(R)
        return R

    def adjustWeightsWithR(self,R):
        self.weights += self.studyspeed * R *  np.transpose(self.inputs)

    def adjustThresholdWithR(self,R):
        self.threshold += -1* self.studyspeed * R

    def getThreshold(self):
        return self.threshold

class Neuron_hide():
    def __init__(self,nInput = 0,studySpeed = 1,thresholdFunc = sigmoidX):
        self.nInput = nInput
        #使用随机数初始化权值
        randomWeights = [random.uniform(-1, 1) for i in range(nInput)]
        randomThreshold = random.uniform(0, 1)
        self.weights = randomWeights
        self.threshold = randomThreshold
        self.thresholdFunc = thresholdFunc
        self.studyspeed = studySpeed
        self.inputs = []

    def setInputs(self,inputs):
        self.inputs = inputs

    def getInputs(self):
        return self.inputs

    def getOutput(self,inputs):
        self.inputs = inputs
        output = np.squeeze(np.asarray(self.weights * self.inputs))
        return round(self.thresholdFunc(output-self.threshold),3)

    def getWeights(self):
        return np.squeeze(np.asarray(self.weights))

    def getWeightsMatrix(self):
        return self.weights

    def setWeights(self,weights):
        self.weights = np.matrix(weights)

    def setStudySpeed(self,studyspeed):
        self.studyspeed = studyspeed

    def getThreshold(self):
        return self.threshold

    def getR(self,inputs,sum,output = None):
        if output == None:
            f = self.getOutput(inputs)
        else:
            f = output
        R = f * (1 - f) * sum
        self.adjustWeightsWithR(R)
        return R

    def adjustWeightsWithR(self,R):
        self.weights += self.studyspeed * R *  np.transpose(self.inputs)

    def adjustThresholdWithR(self,R):
        self.threshold += -1* self.studyspeed * R

class NeuralNetWork():
    def __init__(self,nInput=0,nHide=0,nOutput=0,studyspeed=0):
        self.nInput = nInput
        self.nHide = nHide
        self.nOutput = nOutput
        self.studyspeed = studyspeed
        self.inputNeurons = [0 for i in range(nInput)]
        self.hideNeurons = [
        Neuron_hide(nInput = nInput,studySpeed = studyspeed) for i in range(nHide)]
        self.outputNeurons = [
        Neuron_output(nHide = nHide,studySpeed = studyspeed) for i in range(nOutput)]
    
    def backpropagation(self,trainings=[],examples=[]):
        for i in range(len(trainings)):
            self.backpropagationSingle(
                training = trainings[i],
                example = examples[i],
            )

    def backpropagationSingle(self,training,example):
        #用于保存隐藏层输出
        hideOutputs = []
        #用于保存输出层输出
        outputOutputs = []
        #保存调节前输出层权值
        outputWeights_before = []
        #保存调节前隐藏层权值
        #hideWeights_before = []
        #保存调节前输出层阈值
        #outputThresholds_before = []
        #保存调节前隐藏层阈值
        #hideThresholds_before = []
        #保存输出层各个神经元的调节因子
        outputRs = []
        #保存隐藏层各个神经元的调节因子
        #hideRs = []

        #d为训练集目标输出
        d = example
        #输入转化为单列矩阵
        self.inputNeurons = np.transpose(np.matrix(training)/256)

        #对输入层作为输入，得到隐藏层输出结果
        for hideNeuron in self.hideNeurons:
            hideOutputs.append(hideNeuron.getOutput(self.inputNeurons))
        
        #将隐藏层结果作为输出层的输入，得到输出层的结果
        hideOutPutMatrixs =  np.transpose(np.matrix(hideOutputs))
        for outputNeuron in self.outputNeurons:
            outputOutputs.append(outputNeuron.getOutput(hideOutPutMatrixs))

        #对输出层进行调节
        for i in range(self.nOutput):
            f = outputOutputs[i]
            outputR = f * (1 - f) * (f - d[i])*-1
            outputRs.append(outputR)
            outputWeights_before.append(self.outputNeurons[i].getWeights())
            self.outputNeurons[i].adjustWeightsWithR(outputR)
            self.outputNeurons[i].adjustThresholdWithR(outputR)
        
        #对隐藏层进行调节
        for i in range(self.nHide):
            sumOutputRW=0
            for j in range(self.nOutput):
                sumOutputRW += outputRs[j] * outputWeights_before[j][i]
            hideR = hideOutputs[i] * (1 - hideOutputs[i]) * sumOutputRW
            self.hideNeurons[i].adjustWeightsWithR(hideR)
            self.hideNeurons[i].adjustThresholdWithR(hideR)

    def getOutput(self,inputs):
        hideoutputs = []
        inputMatrix = np.transpose(np.matrix(inputs)/256)
        #一输入层为输入计算每个隐藏层神经元输出，得到隐藏层输出结果
        for hideNeuron in self.hideNeurons:
            hideoutputs.append(hideNeuron.getOutput(inputMatrix))

        outputLayeroutputs = []
        #将隐藏层结果作为输出层的输入，得到输出层的结果
        hideoutpuMatrix = np.transpose(np.matrix(hideoutputs))
        for outputNeuron in self.outputNeurons:
            outputLayeroutputs.append(outputNeuron.getOutput(hideoutpuMatrix))
        return outputLayeroutputs


    def getHideNeurons(self):
        return self.hideNeurons

    def setHideNeurons(self,arg):
        self.hideNeurons = arg

    def getOutputNeurons(self):
        return self.outputNeurons

    def setOutputNeurons(self,arg):
        self.outputNeurons = arg

    def getSaveHideNeurons(self):
        res = []
        info={}
        for hideNeuron in self.getHideNeurons():
            info['Threshold']=hideNeuron.getThreshold()
            info['Weights']= hideNeuron.getWeights().tolist()
            res.append(info)
        return res

    def getSaveOutPutNeurons(self):
        res = []
        info={}
        for outputNeuron in self.getOutputNeurons():
            info['Threshold']=outputNeuron.getThreshold()
            info['Weights']= outputNeuron.getWeights().tolist()
            res.append(info)
        return res

    def getSaveData(self):
        saveData = {}
        saveData['nInput'] = self.nInput
        saveData['nHide'] = self.nHide
        saveData['nOutput'] = self.nOutput
        saveData['studyspeed'] = self.studyspeed
        saveData['hideNeurons'] = self.getSaveHideNeurons()
        saveData['ouputNeurons'] = self.getSaveOutPutNeurons()
        return saveData

    # def setBySaveData(self,savedata):
    #     self.nInput = savedata['nInput']
    #     self.nHide = savedata['nHide']
    #     self.nOutput = savedata['nOutput']
    #     self.studyspeed = savedata['studyspeed']
    #     self.inputNeurons = [0 for i in range(nInput)]
    #     self.hideNeurons = [
    #     Neuron(nInput = nInput,studyspeed = studyspeed) for i in range(nHide)]
    #     self.outputNeurons = [
    #     Neuron(nInput = nHide,studyspeed = studyspeed) for i in range(nOutput)]
    #     #设定隐藏层权重
    #     hideDatas = savedata['hideNeurons']
    #     for i in range(nHide):
    #         self.hideNeurons[i].setWeights(np.matrix(hideDatas[i]))
    #     outputDatas = savedata['ouputNeurons']
    #     for i in range(nOutput):
    #         self.outputNeurons[i].setWeights(np.matrix(outputDatas[i]))

    def printInfo(self):
        print('hide')
        for hideNeuron in self.hideNeurons:
            hideNeuron.printInfo()
        print('output')
        for outputNeuron in self.outputNeurons:
            outputNeuron.printInfo()

def trainingFunc(times,nInput,nHide,nOutput):
    try:
        data_file = open('img.json','r')
        res_file = open('res_%d_%d_%d_%d'%(times,nInput,nHide,nOutput),'w+')
        json_data = eval(data_file.read())
        #获得训练集
        imgs = json_data['imgs']
        trainingimgs = imgs[:400]
        testimgs = imgs[400:520]
        print(len(trainingimgs)-120,len(testimgs))
        #定义神经网络
        network = NeuralNetWork(nInput = nInput , nHide = nHide,nOutput = nOutput,studyspeed = 0.05)

		#训练过程
        for time in range(times):
            count = 0
            # print(type(randoms))
            # del randoms[4*i,4*(i+1)]
            trainings=[]
            for j in range(280):
                r=random.randint(0,399)
                #print(r,len(trainingimgs))
                trainings.append(trainingimgs[r])
            for img in trainings:
                example=[0 for _ in range(40)]
                training = img['data']
                example[int(img['name'])]=1
                network.backpropagationSingle(training = training,example = example)
                #print('success training time:%d network:%d item:%d'%(time,i,count),img['name'])

                # count=0
                # success=0
                # for img in testimgs: 
                # 	testItem = img['data']
                # 	if int(img['name'])==i:
                # 		example=1
                # 	else:
                # 		example=0
                # 	res =  network[i].getOutput(testItem)
                # 	if int(res[0]+0.5)==example:
                # 		success+=1
                # 		#print('Trainings times %d network:%d times:%d result:success!'%(time,i,count))
                # 	else:
                # 		print('Trainings times %d network:%d times:%d result:failed!'%(time,i,count),img['name'])
                # 	count+=1
                # input()
                # print('Trainings times %d network:%d : %f'%(time ,i,float(success)/len(testimgs)))
                # res_file.write('[%d,%d,%f]\n'%(time,i,float(success)/len(testimgs)))
            
            count=0
            success=0
            for img in testimgs:
                testItem = img['data']
                name=0
                name=int(img['name'])
                res=network.getOutput(testItem)
                max=0
                for i in range(40):
                    if res[max]<res[i]:
                        max=i
                if max==name:
                    success+=1
                else:
                    print(img['name'],max)
                #max=0
                # for i in range(40):
                # 	if res[i]>res[max]:
                # 		max=i
                # if max!=name:
                # 	success+=1
                    #print('Trainings times %d testid:%d result:success!'%(time,count))
                #else:
                    #print('Trainings times %d testid:%d result:failed!'%(time,count))
                count+=1
            print('times:%d final recognition rate:'%(time),float(success)/count)
            res_file.write('times:%d final recognition rate:%f'%(time,float(success)/count))
            if success==count:
                print("Break?")
                b=input()
                if b!='No':
                    break

        p=str(float(success)/count)
        print('Save training result ?')
        isSave=input()
        if isSave:
            saveNet(network,0,p)


            # count = 0
            # for img in trainingimgs:
            # 	example=[0 for _ in range(40)]
            # 	training = img['data']
            # 	name=int(img['name'])
            # 	example[name-1]=1
            # 	network.backpropagationSingle(training = training,example = example)
            # 	print(count)
            # 	count += 1
            # 	print('success training time:%d item:%d'%(time,count))
            # #测试
            # count = 0
            # #记录预测成功的用例数量
            # success = 0
            # for img in testimgs:
            # 	testItem = img['data']
            # 	name=int(img['name'])
            # 	print('testid',count,':',name)
            # 	res =  network.getOutput(testItem)
            # 	#print(res)
            # 	max=0
            # 	result=-1
            # 	for i in range(40):
            # 		if max<res[i]:
            # 			max=res[i]
            # 			result=i+1
            # 	if result == name:
            # 		success += 1
            # 		print("success",result)
            # 	count += 1
            # print('Trainings times %d : %f'%(time ,float(success)/len(testimgs)))
            # res_file.write('[%d,%f]\n'%(time ,float(success)/len(testimgs)))
    except Exception as e:
	    print(traceback.format_exc())

def saveNet(network,i,p):
    data=network.getSaveData()
    file_name='net'+str(i)+'_'+p+'.json'
    with open(file_name, 'w') as json_file:
        json.dump(data,json_file)


def testOutput(times):
	data_file = open('img.json','r')
	json_data = eval(data_file.read())
	#根据配置文件设置神经网络
	network = NeuralNetWork()
	#获取神经网络配置文件(存储了神经网络的各层神经元数目和各个神经元的权值)
	config_file = open('net%s.json'%times,'r')
	json_config = eval(config_file.read())
	#network.setBySaveData(json_config)
	#获得训练集
	imgs = json_data['imgs']
	for img in imgs:
		testItem = img['data']
		if img['glass'] == 'open':
			example = [1]
		else:
			example = [0]
		print('testid',count,':',example)
		res =  network.getOutput(testItem)
		print(res)
		if round(res[0],0) == example[0]:
			success += 1
		count += 1
	print('success for times %d :'%times ,success/624)

def main():
	#Debug()
	trainingFunc(1000,10304,110,40)
	
if __name__ == '__main__':
	main()
