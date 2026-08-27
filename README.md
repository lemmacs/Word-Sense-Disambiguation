 Christian Lemma

 Title: Word Sense Disambiaguation using glove word embeddings as features

 Description:  This script performs word sense disambiguation on the word "line" from given text files using pre-trained GloVe embeddings
               as input features. It supports two learning models: SVM and NN. It outputs predictions in XML <answer> 
               format to STDOUT, and prepared for scoring.


 User Guide:
   
   - Install GloVe embeddings using this link:
       * https://nlp.stanford.edu/projects/glove/
       * download glove.6B.zip
       * use glove.6B.100d.txt
   - Run the wsd-embeddings.py file using this command:
       * ` python3 wsd-embeddings.py line-train.txt line-test.txt glove.6B.100d.txt [SVM]or[NN] > my-line-answers.txt` 
   - Print out the new answer text file created with this command:
       * `type my-line-answers.txt`



Sample Output:
 
<answer instance="line-n.w8_059:8174:" senseid="product"/>
<answer instance="line-n.w7_098:12684:" senseid="phone"/>
<answer instance="line-n.w8_106:13309:" senseid="phone"/>
<answer instance="line-n.w9_40:10187:" senseid="phone"/>
<answer instance="line-n.w9_16:217:" senseid="phone"/>
<answer instance="line-n.w8_119:16927:" senseid="product"/>

 
