#!/bin/bash

javac -cp jars/stanford-corenlp-3.8.0.jar:jars/edu.mit.jwi_2.4.0.jar::bin/ -d bin/ src/Sentence.java src/Word.java src/Lesk.java 
