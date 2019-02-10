Skip to content
 
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 @yj1399 Sign out
23
252 60 ArunMichaelDsouza/tensorflow-image-detection
 Code  Issues 4  Pull requests 0  Projects 0  Wiki  Insights
tensorflow-image-detection/classify.py
90d5422  on 13 Mar 2018
@ArunMichaelDsouza ArunMichaelDsouza File dialog support added with PR #3
@ArunMichaelDsouza @royalbhati
    
46 lines (34 sloc)  1.43 KB
import tensorflow as tf
import sys
import os
import tkinter as tk
from tkinter import filedialog

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# image_path = sys.argv[1]

root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename()

if image_path:
    
    # Read the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("tf_files/retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
© 2019 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
Press h to open a hovercard with more details.
