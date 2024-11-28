1) First, using the embedder-for-qml.ipynb laptop, we perform preprocessing. Using the SentenceTransformers library, we build informative vector (size 384) representations for our comments. Also in the same laptop, for the subsequent quantum algorithm, we train a classical autoencoder to compress a vector of size 384 into NUM_QUBITS (=4 in this task)


2) Next, in the classical.ipynb notebook, the problem is solved in a classical way based on the received embeddings of proposals. We have built a simple network with a single linear layer and trained it for the classification task


3) The script pyideem-qml-train-test.py We used it to train a quantum variational algorithm (ZZFeatureMap + RealAmplitudeAnsatz) for the binary classification problem on the pyideem backend. The results of its execution are the dependences of the loss function on the number of iterations for the train and test samples (separated by hand), as well as a set of optimal parameters


4) The latest qml.ipynb notebook initializes the quantum chain with the optimal parameters found earlier, draws the loss functions from the number of iterations and calculates the accuracy metric
