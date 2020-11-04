import numpy
import random
import pandas


class NN():

    def specify_structure(self, layernumber, nodesperlayer, linear):

        self.layernumber = layernumber
        self.numberofnodesperlayer = nodesperlayer
        self.nodevalues = []
        self.weightsforward = []
        self.biases = []
        self.defaultbias = 0
        self.linear = linear
        self.desiredepochs = 10000

        standin = []
        for layer in range(layernumber):
            layernodevalues = []
            layerweights = []

            for node in range(nodesperlayer[layer]):
                nodeweights = []
                if not layer + 1 == layernumber:

                    for nextlayersize in range(nodesperlayer[layer + 1]):
                        nodeweights.append(random.gauss(0, 1))
                        # nodeweights.append(random.randint(0,100)/100)
                layerweights.append(nodeweights)
                layernodevalues.append(0)
            standin.append(layernodevalues)
            self.weightsforward.append(layerweights)

            self.biases.append(self.defaultbias)
        self.nodevalues = [x[:] for x in standin]
        self.blanknodes = [x[:] for x in standin]
        self.presquishes = [x[:] for x in standin]

        self.errornodes = [x[:] for x in standin]
        self.blankerrors = [x[:] for x in standin]
        print(self.biases)

    def returnanything(self):
        return (self.weightsforward)

    def train(self, X, y):

        self.X = X.to_numpy()
        self.y = y.to_numpy()

        def sigmoid_crusher(invalue):
            return (1 / (1 + numpy.e ** (-invalue)))

        def multall(inputarray, weightsforward):
            #out = [0 for g in weightsforward[0]]
            out = [0 for g in weightsforward[0]]
            for input in range(len(inputarray)):
                toadd = [inputarray[input] * g for g in weightsforward[input]]
                out = [out[g] + toadd[g] for g in range(len(out))]


            return (out)

        def forwardprop(nodes, weights, biases, inputs, desiredoutputs):

            for layer in range(len(nodes) - 1):
                if layer == 0:
                    self.nodevalues[layer] = inputs

                addup = [x + self.biases[layer] for x in multall(self.nodevalues[layer], self.weightsforward[layer])]
                if layer == len(nodes) - 2:

                    self.nodevalues[layer + 1] = addup
                    self.presquishes[layer + 1] = addup
                else:
                    self.presquishes[layer + 1] = addup
                    self.nodevalues[layer + 1] = [sigmoid_crusher(x) for x in addup]
                # self.nodevalues[layer+1] = [sigmoid_crusher(x + self.biases[layer]) for x in multall(self.nodevalues[layer], self.weightsforward[layer])]
            # alterror = [desiredoutputs[x] - self.nodevalues[-1][x] for x in range()]

            alterror = desiredoutputs - self.nodevalues[-1][0]

            return (alterror)

        def sigma_derivative(output):
            return (output * (1 - output))

        def multback(weights, outputs, error):
            # find errors of previous node layer
            errorslala = [0 for x in outputs]
            for inerror in range(len(error)):
                toadd = [(error[inerror] * weights[x][inerror]) * sigma_derivative(outputs[x]) for x in
                         range(len(weights))]

                errorslala = [toadd[b] + errorslala[b] for b in range(len(toadd))]

            return (errorslala)

        def backpropagation(errors, weights, biases):

            for layerbackwards in range(1, len(self.errornodes)).__reversed__():

                if layerbackwards == len(self.errornodes) - 1:
                    self.errornodes[layerbackwards] = errors


                self.errornodes[layerbackwards - 1] = multback(self.weightsforward[layerbackwards - 1],
                                                               self.nodevalues[layerbackwards - 1],
                                                               self.errornodes[layerbackwards])


        # def update_weights():

        # erroror = forwardprop(self.nodevalues, self.weightsforward, self.biases, self.X[1], self.y[1])

        # print(backpropagation([erroror], self.weightsforward, self.biases))
        # print(self.errornodes)
        # print("error" + str(self.errornodes[-1]))

        def adjust_weights():
            LR = 0.1

            for layer in range(1, len(self.presquishes)):
                # print("ERROR")
                # print(self.errornodes)
                # print("PRE")
                # print(self.presquishes)
                for weightbatch in range(len(self.weightsforward[layer - 1])):
                    self.weightsforward[layer - 1][weightbatch] = [
                        self.weightsforward[layer - 1][weightbatch][ind] + LR * self.errornodes[layer][ind] *
                        self.nodevalues[layer - 1][weightbatch] for ind in
                        range(len(self.weightsforward[layer - 1][weightbatch]))]

                    # self.biases[layer - 1] = [self.biases[weightbatch] + LR*self.errornodes[layer][weightbatch]]
                    # self.biases[layer - 1] += LR*self.errornodes[layer][weightbatch]*self.presquishes[layer-1][weightbatch]
                #self.biases[layer - 1] += sum(LR*self.errornodes[layer][ind]*self.presquishes[layer][ind] for ind in range(len(self.errornodes[layer])))

        # adjust_weights()
        # self.errornodes = [x[:] for x in self.blankerrors]
        # self.nodevalues = [x[:] for x in self.blanknodes]
        # onglo = forwardprop(self.nodevalues, self.weightsforward, self.biases, self.X[1], self.y[1])
        # backpropagation([onglo], self.weightsforward, self.biases)
        # print("error" + str(self.errornodes[-1]))

        while True:
            break
            errorrr = forwardprop(self.nodevalues, self.weightsforward, self.biases, self.X[2], self.y[2])
            backpropagation([errorrr], self.weightsforward, self.biases)
            # print("error" + str(self.errornodes[-1]))
            # print(self.errornodes)
            #print(self.errornodes[-1][0])
            print("")
            print("")
            print("")
            print("errors")
            print(self.errornodes)
            print("")
            print("nodes")
            print(self.nodevalues)
            print("")
            print("weights")
            print(self.weightsforward)
            print("")
            print("")
            print("")

            # print(self.weightsforward)

            adjust_weights()
            self.errornodes = [x[:] for x in self.blankerrors]
            self.nodevalues = [x[:] for x in self.blanknodes]


        for epoch in range(self.desiredepochs):

            defaultnodeweights = [x[:] for x in self.weightsforward]

            weightchanges = []
            epocherror = 0
            for element in range(len(self.X)):
                errorrr = forwardprop(self.nodevalues, self.weightsforward, self.biases, self.X[element],
                                      self.y[element])
                epocherror += abs(errorrr)
                backpropagation([errorrr], self.weightsforward, self.biases)
                adjust_weights()
                weightchanges.append(self.weightsforward)
                self.errornodes = [x[:] for x in self.blankerrors]
                self.nodevalues = [x[:] for x in self.blanknodes]
                self.weightsforward = [x[:] for x in defaultnodeweights]
            #print(self.weightsforward)
            #print(len(self.weightsforward))
            for layer in range(len(self.weightsforward)):

                for weightbatch in range(len(self.weightsforward[layer])):

                    for individualweight in range(len(self.weightsforward[layer][weightbatch])):
                        #print(self.weightsforward[layer][nodeweightbatch][individualweight])
                        self.weightsforward[layer][weightbatch][individualweight] = numpy.mean([weightchanges[x][layer][weightbatch][individualweight] for x in range(len(weightchanges))])


            print(epocherror)





        while True:
            want_another_test = input("Test_again?: ")
            if want_another_test == "n":
                break
            testdatapoint = random.randint(0,len(self.X) - 1)
            print("desired_output " + str(self.y[testdatapoint]))
            forwardprop(self.nodevalues, self.weightsforward, self.biases, self.X[testdatapoint], self.y[testdatapoint])

            print(self.nodevalues[-1])

            #print(self.weightsforward)
            self.nodevalues = [x[:] for x in self.blanknodes]



#animaldata = pandas.read_csv(r"C:\Users\dillo\Documents\Anaconda_files\Animalqualities2.csv")
animaldata = pandas.read_csv(r"C:\Users\dillo\Documents\Anaconda_files\iris.csv")
X = animaldata.drop(columns=['petal.width', 'variety'])
y = animaldata['petal.width']
#X = animaldata.drop(columns=['Teeth'])
#y = animaldata['Teeth']

folio = NN()
#folio.specify_structure(10, [3, 8,8 , 8, 8, 8, 8, 8, 8, 1], True)
folio.specify_structure(3, [3, 6, 1], False)
folio.train(X, y)